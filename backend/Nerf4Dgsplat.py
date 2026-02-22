# Nerf4Dgsplat.py — Video-to-Gaussian-Splat Pipeline
# Extracts frames from videos, estimates camera poses via COLMAP,
# trains a Gaussian Splatting model using nerfstudio's Splatfacto,
# and exports the result as a standard .ply file.
#
# Usage:
#   python Nerf4Dgsplat.py
#
# Requirements:
#   pip install nerfstudio plyfile

from __future__ import annotations

import json
import subprocess
import os
from collections import OrderedDict
import argparse
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
from plyfile import PlyData, PlyElement

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, SH2RGB
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.utils.eval_utils import eval_setup


# ---------------------------------------------------------------------------
# 1. TRAINING
# ---------------------------------------------------------------------------

def run_gs_reconstruction(
    data_dir: str,
    output_dir: str = "outputs/gs",
    iterations: int = 5000,
) -> tuple[Path, SplatfactoModel]:
    """
    Trains a Gaussian Splatting model (Splatfacto) on a prepared nerfstudio dataset.

    Args:
        data_dir: Path to a directory with transforms.json + images/ (nerfstudio format).
        output_dir: Where to save checkpoints and config.
        iterations: Number of training steps.

    Returns:
        (config_yaml_path, trained_model)
    """
    config = TrainerConfig(
        method_name="splatfacto",
        max_num_iterations=iterations,
        steps_per_save=1000,
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    data=Path(data_dir),
                    load_3D_points=True,
                ),
                cache_images="gpu",
                cache_images_type="uint8",
            ),
            model=SplatfactoModelConfig(
                stop_split_at=min(4000, iterations),
                cull_alpha_thresh=0.05,
                densify_grad_thresh=0.0005,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                sh_degree=2,
                num_downscales=2,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=1.0e-3, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15),
                "scheduler": None,
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=1.25e-4, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-15),
                "scheduler": None,
            },
        },
        vis="tensorboard",  # headless-compatible; no viewer server needed
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        output_dir=Path(output_dir),
        mixed_precision=True,
    )

    # --- Critical: setup() creates pipeline, optimizers, etc. ---
    trainer = config.setup()
    trainer.setup()

    # Save config.yml NOW (before training, so it's always written even if training crashes)
    config.save_config()

    trainer.train()

    config_yaml = trainer.base_dir / "config.yml"
    ckpt_dir = trainer.checkpoint_dir
    print(f"✅ Training complete.")
    print(f"   Config:      {config_yaml}")
    print(f"   Checkpoints: {ckpt_dir}")

    return config_yaml, trainer.pipeline.model


# ---------------------------------------------------------------------------
# 2. EXPORT DIRECTLY FROM CHECKPOINT (no config.yml needed)
# ---------------------------------------------------------------------------

def export_from_checkpoint(
    ckpt_path: str | Path,
    out_rgb: str | Path = "./splat_rgb.ply",
    out_sh: str | Path = "./splat_sh.ply",
) -> None:
    """
    Re-exports a .ply from a raw nerfstudio checkpoint without needing config.yml.
    Produces two files:
      - RGB PLY:  works in ALL viewers (supersplat, antimatter15, etc.)
      - SH PLY:   higher quality in viewers that support spherical harmonics

    Args:
        ckpt_path: Path to a step-XXXXXX.ckpt file.
        out_rgb: Output path for the RGB PLY.
        out_sh:  Output path for the SH PLY.
    """
    C0 = 0.28209479177387814  # SH DC normalization constant

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    p = ckpt["pipeline"]

    means   = p["_model.gauss_params.means"].float().numpy()
    f_dc    = p["_model.gauss_params.features_dc"].float().numpy()   # [N,3]
    f_rest  = p["_model.gauss_params.features_rest"].float().numpy() # [N,K,3]
    opacs   = p["_model.gauss_params.opacities"].float().numpy()     # [N,1]
    scales  = p["_model.gauss_params.scales"].float().numpy()        # [N,3]
    quats   = p["_model.gauss_params.quats"].float().numpy()         # [N,4]

    N = means.shape[0]
    colors_01 = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)  # SH DC -> linear RGB

    # Filter NaN/Inf and near-invisible Gaussians
    select = (
        np.isfinite(means).all(axis=1) &
        np.isfinite(colors_01).all(axis=1) &
        (opacs.squeeze() >= -5.5373)  # logit(1/255)
    )
    n = int(select.sum())
    removed = N - n
    if removed:
        print(f"  Filtered {removed:,}/{N:,} NaN/low-opacity Gaussians")
    print(f"  Exporting {n:,} Gaussians")

    means_f  = means[select].astype(np.float32)
    f_dc_f   = f_dc[select].astype(np.float32)
    f_rest_f = f_rest[select].astype(np.float32)
    opacs_f  = opacs[select].squeeze().astype(np.float32)
    scales_f = scales[select].astype(np.float32)
    quats_f  = quats[select].astype(np.float32)
    colors_u8 = (colors_01[select] * 255).astype(np.uint8)

    def _build_common_header(m):
        m["x"] = means_f[:, 0]; m["y"] = means_f[:, 1]; m["z"] = means_f[:, 2]
        m["nx"] = np.zeros(n, np.float32)
        m["ny"] = np.zeros(n, np.float32)
        m["nz"] = np.zeros(n, np.float32)

    def _build_common_tail(m):
        m["opacity"] = opacs_f
        for i in range(3): m[f"scale_{i}"] = scales_f[:, i]
        for i in range(4): m[f"rot_{i}"]   = quats_f[:, i]

    # RGB PLY
    rgb_map: OrderedDict[str, np.ndarray] = OrderedDict()
    _build_common_header(rgb_map)
    rgb_map["red"] = colors_u8[:, 0]
    rgb_map["green"] = colors_u8[:, 1]
    rgb_map["blue"]  = colors_u8[:, 2]
    _build_common_tail(rgb_map)
    Path(out_rgb).parent.mkdir(parents=True, exist_ok=True)
    _write_gs_ply(str(out_rgb), n, rgb_map)
    print(f"✅ RGB PLY -> {out_rgb}")

    # SH PLY
    sh_map: OrderedDict[str, np.ndarray] = OrderedDict()
    _build_common_header(sh_map)
    for i in range(3): sh_map[f"f_dc_{i}"] = f_dc_f[:, i]
    # Transpose [N,K,3] -> [N,3,K] -> flatten [N, K*3], matching inria format
    shs_rest_t = f_rest_f.transpose(0, 2, 1).reshape(n, -1)
    for i in range(shs_rest_t.shape[1]): sh_map[f"f_rest_{i}"] = shs_rest_t[:, i]
    _build_common_tail(sh_map)
    Path(out_sh).parent.mkdir(parents=True, exist_ok=True)
    _write_gs_ply(str(out_sh), n, sh_map)
    print(f"✅ SH  PLY -> {out_sh}")


# ---------------------------------------------------------------------------
# 3. MODEL LOADING
# ---------------------------------------------------------------------------

def load_model_into_memory(config_path: str | Path):
    """
    Loads a trained model checkpoint from a config.yml path.

    Returns:
        (pipeline, model)
    """
    config_path = Path(config_path)
    _, pipeline, checkpoint_path, step = eval_setup(
        config_path,
        test_mode="inference",
    )
    print(f"✅ Loaded model from {checkpoint_path} at step {step}")

    model = pipeline.model
    model.eval()
    return pipeline, model


# ---------------------------------------------------------------------------
# 3. DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_datadir(
    video_configs: list[dict],
    output_dir: str = "data_prepared",
    fps: int = 2,
) -> str:
    """
    Extracts frames from video(s), runs COLMAP for camera poses,
    and injects temporal metadata into transforms.json.

    Args:
        video_configs: List of dicts with keys 'path', 'start_sec', 'end_sec'.
        output_dir: Where to write the nerfstudio-format dataset.
        fps: Frames per second to extract from each video.

    Returns:
        Path to the prepared data directory.
    """
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # Timeline info
    all_starts = [v['start_sec'] for v in video_configs]
    all_ends = [v['end_sec'] for v in video_configs]
    global_start = min(all_starts)
    global_duration = max(all_ends) - global_start
    if global_duration <= 0:
        global_duration = 1.0  # avoid division by zero

    # Extraction folder
    extracted_path = output_path / "extracted_images"
    if extracted_path.exists():
        for f in extracted_path.glob("*.jpg"):
            f.unlink()
    if images_path.exists():
        for f in images_path.glob("*.jpg"):
            f.unlink()
    extracted_path.mkdir(parents=True, exist_ok=True)

    print("--- Extracting Frames ---")
    for i, v in enumerate(video_configs):
        v_path = Path(v['path'])
        duration = v['end_sec'] - v['start_sec']
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(v['start_sec']),
            "-t", str(duration),
            "-i", str(v_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(extracted_path / f"vid_{i}_%04d.jpg"),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    # Run COLMAP pose estimation via nerfstudio
    print("--- Running COLMAP (this may take a while) ---")
    subprocess.run([
        "/opt/miniforge3/envs/nerf/bin/ns-process-data", "images",
        "--data", str(extracted_path),
        "--output-dir", str(output_path),
        "--matcher-type", "any",
        "--sfm-tool", "colmap",
        "--no-verbose",
        "--no-gpu",
    ], check=True)

    # Inject temporal metadata into transforms.json
    transforms_file = output_path / "transforms.json"
    with open(transforms_file, "r") as f:
        data = json.load(f)

    print("--- Injecting Temporal Data into transforms.json ---")

    orig_images_sorted = sorted([f.name for f in extracted_path.glob("vid_*.jpg")])

    frame_name_to_info = {}
    for idx, orig_name in enumerate(orig_images_sorted):
        processed_frame_name = f"frame_{idx + 1:05d}.jpg"

        parts = orig_name.split("_")
        vid_idx = int(parts[1])
        frame_num = int(parts[2].split(".")[0])

        local_time_sec = (frame_num - 1) / fps
        abs_time = video_configs[vid_idx]['start_sec'] + local_time_sec
        normalized_time = (abs_time - global_start) / global_duration

        frame_name_to_info[processed_frame_name] = {
            "time": round(float(normalized_time), 4),
            "video_id": vid_idx,
        }

    for frame in data["frames"]:
        file_name = Path(frame["file_path"]).name
        if file_name in frame_name_to_info:
            info = frame_name_to_info[file_name]
            frame["time"] = info["time"]
            frame["video_id"] = info["video_id"]
        else:
            print(f"  Warning: Could not find temporal info for {file_name}")

    with open(transforms_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"--- Done! data_dir is ready at: {output_dir} ---")
    return str(output_path)


# ---------------------------------------------------------------------------
# 4. FULL PIPELINE
# ---------------------------------------------------------------------------

def videos_to_splat(
    meta_data: list[dict],
    output_dir: str = "./",
    fps: int = 2,
    iterations: int = 5000,
) -> tuple[Path, SplatfactoModel]:
    """
    End-to-end: videos -> data preparation -> training -> model.

    Returns:
        (config_yaml_path, trained_model)
    """
    data_dir = prepare_datadir(meta_data, output_dir=os.path.join(output_dir, "data_prepared"), fps=fps)
    config_path, model = run_gs_reconstruction(
        data_dir, output_dir=output_dir, iterations=iterations,
    )
    return config_path, model


# ---------------------------------------------------------------------------
# 5. SPLAT DATA EXTRACTION
# ---------------------------------------------------------------------------

def get_splat_data(model: SplatfactoModel) -> dict[str, torch.Tensor]:
    """
    Extracts the Gaussian Splatting parameters from a trained model.

    Returns a dict with keys: xyz, colors, opacities, scales, quats.
    All tensors are detached and on CPU.
    """
    model.eval()
    with torch.no_grad():
        return {
            "xyz": model.means.detach().cpu(),
            "colors": model.colors.detach().cpu(),       # RGB in [0, 1]
            "opacities": model.opacities.detach().cpu(),  # raw logits
            "scales": model.scales.detach().cpu(),         # raw log-scales
            "quats": model.quats.detach().cpu(),
        }


# ---------------------------------------------------------------------------
# 6. PLY EXPORT   (compatible with standard GS viewers)
# ---------------------------------------------------------------------------

def export_splat_to_ply(
    model: SplatfactoModel,
    output_path: str | Path = "./splat.ply",
    color_mode: str = "sh_coeffs",
) -> None:
    """
    Exports a trained Splatfacto model to a standard Gaussian Splatting .ply file.
    This follows the same format as nerfstudio's built-in exporter, ensuring
    compatibility with web viewers (e.g. antimatter15, playcanvas, etc.).

    Args:
        model: A trained SplatfactoModel instance.
        output_path: Where to save the .ply file.
        color_mode: "sh_coeffs" for full SH data (default), or "rgb" for simple RGB.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    map_to_tensors: OrderedDict[str, np.ndarray] = OrderedDict()

    with torch.no_grad():
        positions = model.means.cpu().numpy()
        n = positions.shape[0]

        # Positions
        map_to_tensors["x"] = positions[:, 0].astype(np.float32)
        map_to_tensors["y"] = positions[:, 1].astype(np.float32)
        map_to_tensors["z"] = positions[:, 2].astype(np.float32)

        # Normals (zeros — standard for GS PLY)
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        # Colors
        if color_mode == "rgb":
            colors = torch.clamp(model.colors, 0.0, 1.0).cpu().numpy()
            colors_u8 = (colors * 255).astype(np.uint8)
            map_to_tensors["red"] = colors_u8[:, 0]
            map_to_tensors["green"] = colors_u8[:, 1]
            map_to_tensors["blue"] = colors_u8[:, 2]
        else:
            # Spherical harmonics — DC component
            shs_0 = model.shs_0.contiguous().cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None].astype(np.float32)

            # Higher-order SH bands
            if model.config.sh_degree > 0:
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None].astype(np.float32)

        # Opacity (raw logits, same as nerfstudio export)
        map_to_tensors["opacity"] = model.opacities.data.cpu().numpy().astype(np.float32)

        # Scales (raw log-scales)
        scales = model.scales.data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None].astype(np.float32)

        # Quaternions
        quats = model.quats.data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None].astype(np.float32)

    # Filter NaN/Inf values
    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        select &= np.isfinite(t).reshape(n) if t.ndim > 1 else np.isfinite(t)

    # Filter near-invisible Gaussians (opacity logit < logit(1/255) ≈ -5.54)
    opa = map_to_tensors["opacity"].squeeze()
    select &= opa >= -5.5373

    filtered_count = n - int(select.sum())
    if filtered_count > 0:
        print(f"  Filtered {filtered_count}/{n} Gaussians (NaN/Inf/low-opacity)")
        for k in map_to_tensors:
            map_to_tensors[k] = map_to_tensors[k][select]
        n = int(select.sum())

    # Write PLY (binary format, compatible with standard GS viewers)
    _write_gs_ply(str(output_path), n, map_to_tensors)
    print(f"✅ Exported {n} Gaussians to {output_path}")


def _write_gs_ply(
    filename: str,
    count: int,
    map_to_tensors: OrderedDict[str, np.ndarray],
) -> None:
    """Write a binary PLY file in standard Gaussian Splatting format."""
    with open(filename, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"comment Generated by Nerf4Dgsplat\n")
        f.write(f"element vertex {count}\n".encode())

        for key, tensor in map_to_tensors.items():
            dtype_str = "float" if tensor.dtype.kind == "f" else "uchar"
            f.write(f"property {dtype_str} {key}\n".encode())

        f.write(b"end_header\n")

        for i in range(count):
            for tensor in map_to_tensors.values():
                val = tensor[i]
                if tensor.dtype.kind == "f":
                    f.write(np.float32(val).tobytes())
                elif tensor.dtype == np.uint8:
                    f.write(val.tobytes())


# ---------------------------------------------------------------------------
# 7. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This section is now handled by main() if called as a script.
    pass

def main():
    parser = argparse.ArgumentParser(description="Nerf4Dgsplat - Video to Gaussian Splat Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="./", help="Output directory for results")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract from video")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of training iterations")
    parser.add_argument("--start_sec", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end_sec", type=float, help="End time in seconds (defaults to video end)")

    args = parser.parse_args()

    # Determine video duration if end_sec not provided
    if args.end_sec is None:
        import cv2
        cap = cv2.VideoCapture(args.video)
        args.end_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    video_configs = [{
        "path": args.video,
        "start_sec": args.start_sec,
        "end_sec": args.end_sec,
    }]

    # Run the full pipeline
    config_path, model = videos_to_splat(
        video_configs,
        output_dir=args.output_dir,
        fps=args.fps,
        iterations=args.iterations,
    )

    # Export results
    baked_path = Path(args.output_dir) / "baked_splat.ply"
    baked_sh_path = Path(args.output_dir) / "baked_splat_sh.ply"
    
    export_splat_to_ply(model, baked_path, color_mode="rgb")
    export_splat_to_ply(model, baked_sh_path, color_mode="sh_coeffs")

    print(f"\n🎉 Pipeline complete!")
    print(f"   Config:       {config_path}")
    print(f"   PLY (RGB):    {baked_path}")
    print(f"   PLY (SH):     {baked_sh_path}")

if __name__ == "__main__":
    main()
