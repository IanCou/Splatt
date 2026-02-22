# Nerf4Dgsplat.py — Video-to-Gaussian-Splat Pipeline
# =============================================================================
# Extracts frames from one or more videos, estimates camera poses via COLMAP,
# trains a Gaussian Splatting model using nerfstudio's Splatfacto, and exports
# the result as a standard .ply file compatible with all major GS viewers.
#
# Usage (CLI):
#   python Nerf4Dgsplat.py --video ./IMG_0040.MOV --fps 5 --iterations 30000
#   python Nerf4Dgsplat.py --video ./clip.mp4 --export_only --ckpt_path ./outputs/gs/.../step-030000.ckpt
#
# Programmatic usage (import as module):
#   from Nerf4Dgsplat import videos_to_splat, export_splat_to_ply
#   config_path, model = videos_to_splat([{"path": "vid.mp4", "start_sec": 0, "end_sec": 12}])
#   export_splat_to_ply(model, "./baked.ply", color_mode="rgb")
#
# Requirements:
#   pip install nerfstudio plyfile
#
#   For editing: SplatfactoModelConfig
# =============================================================================

from __future__ import annotations

import json
import struct
import subprocess
import os
from collections import OrderedDict
import argparse
import uuid
import requests
from pathlib import Path
from typing import Optional, List, Tuple

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
# Section 1 — TRAINING
# ---------------------------------------------------------------------------

def run_gs_reconstruction(
    data_dir: str,
    output_dir: str = "outputs/gs",
    iterations: int = 30000,
) -> Tuple[Path, SplatfactoModel]:
    """
    Train a Gaussian Splatting model (Splatfacto) on a prepared nerfstudio dataset.

    This sets up a full TrainerConfig with production-quality hyperparameters and
    runs the training loop. The config.yml is saved *before* training begins so it
    always exists even if training is interrupted (allowing checkpoint re-export later).

    Args:
        data_dir:   Path to a directory containing transforms.json + images/
                    in nerfstudio format (produced by `prepare_datadir` or
                    `ns-process-data`).
        output_dir: Root directory where checkpoints and config.yml are saved.
                    Subdirectories (e.g. splatfacto/<timestamp>/) are auto-created
                    by nerfstudio.
        iterations: Total training steps. 30 000 is the recommended default for
                    production quality. Use 10 000 for quick previews only.

    Returns:
        Tuple of:
          - config_yaml (Path): Absolute path to the saved config.yml, which can be
                                passed to `load_model_into_memory` later.
          - model (SplatfactoModel): The trained model, still on GPU (or CPU if no GPU).
    """
    config = TrainerConfig(
        method_name="splatfacto",
        max_num_iterations=iterations,
        steps_per_save=2000,            # Save a checkpoint every 2 000 steps for resume/re-export.
        steps_per_eval_image=1000,      # Log an eval render to tensorboard every 1 000 steps.
        steps_per_eval_batch=1000,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    data=Path(data_dir),
                    load_3D_points=True,   # Feeds COLMAP sparse points as initial Gaussian
                                           # positions — dramatically improves convergence.
                ),
                cache_images="gpu",        # Keep all training images on GPU to avoid CPU↔GPU
                cache_images_type="uint8", # I/O bottleneck. uint8 halves VRAM vs float32.
            ),
            model=SplatfactoModelConfig(
                # stop_split_at: Allow densification (splitting/cloning) to run for 80% of
                # training. This gives the model much more time to fill in fine detail —
                # the original 10K cap was too aggressive for complex scenes.
                stop_split_at=max(int(iterations * 0.8), 15000),

                # cull_alpha_thresh: Gaussians with opacity (after sigmoid) below this
                # value are pruned. 0.1 is more aggressive than the old 0.05 — it removes
                # semi-transparent noise splats that contribute to haze/fog artifacts.
                cull_alpha_thresh=0.1,

                # densify_grad_thresh: Split/clone when 2D screen-space gradient exceeds
                # this threshold. 0.0002 (the original 3DGS paper value) produces ~2×
                # more densification than 0.0004, yielding finer geometric detail.
                densify_grad_thresh=0.0002,

                # use_scale_regularization: Penalises Gaussians whose longest axis is
                # much larger than their shortest (high anisotropy ratio). This prevents
                # the thin, stretched floater artifacts that are hard to cull geometrically.
                use_scale_regularization=True,
                max_gauss_ratio=5.0,
                # max_gauss_ratio: Maximum allowed ratio between the longest and shortest
                # scale axes. 5.0 (down from 10) more aggressively penalises elongated
                # splats that appear as needle/disc artifacts.

                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                # SO3xR3: jointly refine rotation (SO3) and translation (R3) for each camera.
                # This corrects imperfect COLMAP poses during training.

                sh_degree=3,
                # Degree 3 spherical harmonics: 16 coefficients per channel (up from 9).
                # Captures finer view-dependent effects (specular highlights, metallic
                # reflections). Costs ~30% more VRAM but significantly improves realism.

                num_downscales=0,
                # Train at full resolution from step 0. With enough iterations (30K)
                # there's no need for a multi-resolution warmup — the model converges
                # to sharper detail when it sees full-res images throughout training.
            ),
        ),
        optimizers={
            # Gaussian positions — with exponential decay (100× reduction over training).
            # Starts broad to let densification place Gaussians, then fine-tunes positions.
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=iterations,
                ),
            },
            # Quaternion rotations — decay to prevent jitter in the final output.
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=1.0e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.0e-5,
                    max_steps=iterations,
                ),
            },
            # Log-scales — moderate decay to stabilise Gaussian sizes late in training.
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=iterations,
                ),
            },
            # Opacity logits — highest LR → rapid fade-out of unused Gaussians.
            # No decay here: the model should always be able to kill bad splats.
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15),
                "scheduler": None,
            },
            # SH DC (base color) — slight decay to lock in colour late in training.
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2.5e-4,
                    max_steps=iterations,
                ),
            },
            # SH higher-order — lower LR + decay to prevent SH overfitting.
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=1.25e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.25e-5,
                    max_steps=iterations,
                ),
            },
            # Camera pose refinement — slow start, decays so poses are frozen by end.
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6,
                    max_steps=iterations,
                ),
            },
        },
        vis="tensorboard",     # "tensorboard" is headless-safe (no HTTP server spun up).
                               # View logs with: tensorboard --logdir <output_dir>
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            quit_on_train_completion=True,  # Don't block after training ends.
        ),
        output_dir=Path(output_dir),
        mixed_precision=True,  # bf16/fp16 mixed precision halves VRAM usage and speeds
                               # up training on Ampere+ GPUs with negligible quality loss.
    )

    # setup() instantiates the pipeline, dataloaders, optimizers, and schedulers.
    trainer = config.setup()
    trainer.setup()

    # Save config.yml *before* training — safeguards against crash mid-run by ensuring
    # the config always exists alongside any checkpoints that get written.
    config.save_config()

    print(f"\n🚀 Starting Splatfacto training for {iterations:,} iterations...")
    print(f"   Data dir:    {data_dir}")
    print(f"   Output dir:  {output_dir}")
    trainer.train()

    config_yaml = trainer.base_dir / "config.yml"
    ckpt_dir = trainer.checkpoint_dir
    print(f"\n✅ Training complete.")
    print(f"   Config:      {config_yaml}")
    print(f"   Checkpoints: {ckpt_dir}")

    return config_yaml, trainer.pipeline.model


# ---------------------------------------------------------------------------
# Section 2 — EXPORT DIRECTLY FROM CHECKPOINT  (config.yml not required)
# ---------------------------------------------------------------------------

def export_from_checkpoint(
    ckpt_path: str | Path,
    out_rgb: str | Path = "./splat_rgb.ply",
    out_sh: str | Path = "./splat_sh.ply",
) -> None:
    """
    Re-export .ply files directly from a raw nerfstudio checkpoint, without
    needing a config.yml. Useful when:
      - training crashed after saving a checkpoint but before saving config.yml
      - you want to re-export a mid-training checkpoint at a specific step
      - you have a checkpoint from an older run and no longer have the config

    Produces two output files:
      - RGB PLY  — baked sRGB colours; works in *all* viewers (supersplat,
                   antimatter15, playcanvas, etc.)
      - SH PLY   — full spherical-harmonic coefficients; higher visual quality
                   in viewers that natively support SH (e.g. inria, luma, etc.)

    Args:
        ckpt_path: Path to a nerfstudio step-XXXXXX.ckpt file.
        out_rgb:   Output path for the RGB .ply file.
        out_sh:    Output path for the SH .ply file.
    """
    # DC-band SH normalization constant (Y_0^0 = 1 / (2*sqrt(π))).
    # The nerfstudio convention stores features_dc as raw SH coefficients,
    # so colour = f_dc * C0 + 0.5 maps them into [0, 1] RGB space.
    C0 = 0.28209479177387814

    print(f"\n🔍 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    p = ckpt["pipeline"]

    # Extract all Gaussian parameters from the checkpoint state dict.
    # All stored as raw (un-activated) values — activations are applied below.
    means   = p["_model.gauss_params.means"].float().numpy()           # [N, 3]  positions
    f_dc    = p["_model.gauss_params.features_dc"].float().numpy()     # [N, 3]  SH DC
    f_rest  = p["_model.gauss_params.features_rest"].float().numpy()   # [N, K, 3] higher SH
    opacs   = p["_model.gauss_params.opacities"].float().numpy()       # [N, 1]  logit opacities
    scales  = p["_model.gauss_params.scales"].float().numpy()          # [N, 3]  log-scales
    quats   = p["_model.gauss_params.quats"].float().numpy()           # [N, 4]  quaternions (wxyz)

    N = means.shape[0]
    print(f"   Total Gaussians in checkpoint: {N:,}")

    # Convert DC SH coefficients to approximate sRGB for filtering and RGB PLY export.
    colors_01 = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)

    # --- Filter out degenerate Gaussians -----------------------------------------
    # logit(1/255) ≈ -5.5373.  Gaussians with opacity logit below this threshold
    # contribute less than 1/255 alpha when rendered and are invisible to viewers.
    # Removing them shrinks the file and speeds up viewer loading with no visual loss.
    select = (
        np.isfinite(means).all(axis=1)       # no NaN/Inf positions
        & np.isfinite(colors_01).all(axis=1)  # no NaN/Inf colours
        & (opacs.squeeze() >= -5.5373)         # opacity > 1/255
    )
    n = int(select.sum())
    removed = N - n
    if removed:
        print(f"   Filtered {removed:,} degenerate Gaussians (NaN / near-invisible)")
    print(f"   Exporting {n:,} Gaussians")

    # Slice and ensure correct dtypes for PLY serialisation.
    means_f   = means[select].astype(np.float32)
    f_dc_f    = f_dc[select].astype(np.float32)
    f_rest_f  = f_rest[select].astype(np.float32)
    opacs_f   = opacs[select].squeeze().astype(np.float32)
    scales_f  = scales[select].astype(np.float32)
    quats_f   = quats[select].astype(np.float32)
    colors_u8 = (colors_01[select] * 255).astype(np.uint8)

    # ---- Shared helper closures -------------------------------------------------
    def _header(m: OrderedDict) -> None:
        """Write xyz + zero normals (standard for GS PLY)."""
        m["x"] = means_f[:, 0]
        m["y"] = means_f[:, 1]
        m["z"] = means_f[:, 2]
        m["nx"] = np.zeros(n, np.float32)
        m["ny"] = np.zeros(n, np.float32)
        m["nz"] = np.zeros(n, np.float32)

    def _tail(m: OrderedDict) -> None:
        """Write raw opacity logit, log-scale, and quaternion fields."""
        m["opacity"] = opacs_f
        for i in range(3):
            m[f"scale_{i}"] = scales_f[:, i]
        for i in range(4):
            m[f"rot_{i}"] = quats_f[:, i]

    # ---- RGB PLY ----------------------------------------------------------------
    rgb_map: OrderedDict[str, np.ndarray] = OrderedDict()
    _header(rgb_map)
    rgb_map["red"]   = colors_u8[:, 0]
    rgb_map["green"] = colors_u8[:, 1]
    rgb_map["blue"]  = colors_u8[:, 2]
    _tail(rgb_map)
    Path(out_rgb).parent.mkdir(parents=True, exist_ok=True)
    _write_gs_ply(str(out_rgb), n, rgb_map)
    print(f"✅ RGB PLY → {out_rgb}")

    # ---- SH PLY -----------------------------------------------------------------
    sh_map: OrderedDict[str, np.ndarray] = OrderedDict()
    _header(sh_map)
    for i in range(3):
        sh_map[f"f_dc_{i}"] = f_dc_f[:, i]   # scalar per-Gaussian, one channel at a time

    # Inria/3DGS PLY expects SH rest coefficients interleaved as [N, num_coeffs * 3]
    # where the inner axis is (coeff_0_R, coeff_0_G, coeff_0_B, coeff_1_R, ...).
    # nerfstudio stores f_rest as [N, K, 3], so we transpose to [N, 3, K] then flatten.
    shs_rest_t = f_rest_f.transpose(0, 2, 1).reshape(n, -1)   # [N, K*3]
    for i in range(shs_rest_t.shape[1]):
        sh_map[f"f_rest_{i}"] = shs_rest_t[:, i]

    _tail(sh_map)
    Path(out_sh).parent.mkdir(parents=True, exist_ok=True)
    _write_gs_ply(str(out_sh), n, sh_map)
    print(f"✅ SH  PLY → {out_sh}")


# ---------------------------------------------------------------------------
# Section 3 — MODEL LOADING
# ---------------------------------------------------------------------------

def load_model_into_memory(
    config_path: str | Path,
) -> Tuple[object, SplatfactoModel]:
    """
    Load a trained Splatfacto model from a saved config.yml into GPU/CPU memory.

    This is the standard nerfstudio path for post-training evaluation and rendering.
    It automatically locates the latest checkpoint in the same directory as config.yml.

    Args:
        config_path: Absolute or relative path to the config.yml saved by
                     `run_gs_reconstruction` (or `ns-train splatfacto`).

    Returns:
        Tuple of:
          - pipeline: The full VanillaPipeline (useful if you need the datamanager
                      or want to run inference through nerfstudio's rendering API).
          - model (SplatfactoModel): The model in eval mode, ready for rendering
                                     or export via `export_splat_to_ply`.
    """
    config_path = Path(config_path)
    _, pipeline, checkpoint_path, step = eval_setup(
        config_path,
        test_mode="inference",
    )
    print(f"✅ Loaded model from {checkpoint_path} at step {step:,}")

    model = pipeline.model
    model.eval()
    return pipeline, model


# ---------------------------------------------------------------------------
# Section 4 — DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_datadir(
    video_configs: list[dict],
    output_dir: str = "data_prepared",
    fps: int = 4,
) -> str:
    """
    Extract frames from one or more videos, run COLMAP to estimate camera poses,
    and inject per-frame temporal metadata into the resulting transforms.json.

    The temporal metadata (`time`, `video_id` fields on each frame) enables
    time-conditioned models (e.g. 4D/dynamic Gaussian Splatting) to associate
    each training image with a point in the scene's timeline.

    Args:
        video_configs: List of dicts, each with:
            - "path"      (str):   Path to the video file.
            - "start_sec" (float): Start time within the video (seconds).
            - "end_sec"   (float): End time within the video (seconds).
        output_dir: Directory where the nerfstudio dataset will be written.
                    Will contain: images/, transforms.json, colmap/, etc.
        fps: Frames per second to extract from each video clip.
             Higher values give COLMAP more views for better pose estimation
             at the cost of more disk space and longer COLMAP runtime.
             4 fps is a good default for handheld walk-around captures.

    Returns:
        The absolute path of the prepared output directory (same as output_dir,
        resolved to an absolute path).
    """
    output_path = Path(output_dir).resolve()
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # --- Compute global timeline -------------------------------------------------
    # We normalise every frame's absolute timestamp to [0, 1] relative to the full
    # multi-video timeline so the `time` field is always in a consistent range.
    all_starts = [v["start_sec"] for v in video_configs]
    all_ends   = [v["end_sec"]   for v in video_configs]
    global_start    = min(all_starts)
    global_duration = max(all_ends) - global_start
    if global_duration <= 0:
        global_duration = 1.0   # guard against zero-length or single-frame input

    # --- Extract frames ----------------------------------------------------------
    # Frames are extracted into a staging directory (extracted_images/) first,
    # then COLMAP / ns-process-data moves/renames them into images/.
    extracted_path = output_path / "extracted_images"
    if extracted_path.exists():
        for f in extracted_path.glob("*.jpg"):
            f.unlink()
    if images_path.exists():
        for f in images_path.glob("*.jpg"):
            f.unlink()
    extracted_path.mkdir(parents=True, exist_ok=True)

    print("\n─── Step 1/3: Extracting Frames ────────────────────────────────────────")
    for i, v in enumerate(video_configs):
        v_path   = Path(v["path"])
        duration = v["end_sec"] - v["start_sec"]
        out_pattern = str(extracted_path / f"vid_{i}_%04d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(v["start_sec"]),   # seek *before* -i for fast input seeking
            "-t",  str(duration),
            "-i",  str(v_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",                   # JPEG quality: 2 = near-lossless (1–31 scale)
            out_pattern,
        ]
        print(f"  Video {i+1}/{len(video_configs)}: {v_path.name}  "
              f"({v['start_sec']}s – {v['end_sec']}s @ {fps} fps)")
        subprocess.run(cmd, check=True, capture_output=True)

    extracted_frames = sorted(extracted_path.glob("vid_*.jpg"))
    print(f"  Extracted {len(extracted_frames):,} frames total.")

    # --- COLMAP pose estimation --------------------------------------------------
    # ns-process-data wraps COLMAP's feature extraction + matching + triangulation
    # pipeline and outputs a nerfstudio-compatible transforms.json.
    print("\n─── Step 2/3: Running COLMAP (may take several minutes) ────────────────")
    # Set QT_QPA_PLATFORM=offscreen to prevent COLMAP from trying to open a
    # GUI on headless servers (crashes with "could not connect to display").
    colmap_env = os.environ.copy()
    colmap_env["QT_QPA_PLATFORM"] = "offscreen"

    colmap_result = subprocess.run([
        "/opt/miniforge3/envs/nerf/bin/ns-process-data", "images",
        "--data",         str(extracted_path),
        "--output-dir",   str(output_path),
        "--matcher-type", "any",
        "--matching-method",  "sequential",
        "--sfm-tool",     "colmap",
        "--no-gpu",       # COLMAP GPU SIFT requires OpenGL, which isn't available
                          # on headless servers without a display/EGL setup.
    ], capture_output=True, text=True, env=colmap_env)

    if colmap_result.returncode != 0:
        import sys
        # Write to stderr so the error propagates through the SSH error channel
        print(f"COLMAP_FAILED (exit {colmap_result.returncode})", file=sys.stderr)
        if colmap_result.stdout:
            print(f"COLMAP STDOUT:\n{colmap_result.stdout[-2000:]}", file=sys.stderr)
        if colmap_result.stderr:
            print(f"COLMAP STDERR:\n{colmap_result.stderr[-2000:]}", file=sys.stderr)
        # Also print to stdout for streaming visibility
        print(f"  ❌ COLMAP / ns-process-data failed (exit {colmap_result.returncode})")
        print(f"  STDOUT: {colmap_result.stdout[-2000:] if colmap_result.stdout else '(empty)'}")
        print(f"  STDERR: {colmap_result.stderr[-2000:] if colmap_result.stderr else '(empty)'}")
        raise subprocess.CalledProcessError(
            colmap_result.returncode,
            colmap_result.args,
            colmap_result.stdout,
            colmap_result.stderr,
        )

    # --- Inject temporal metadata ------------------------------------------------
    print("\n─── Step 3/3: Injecting Temporal Metadata into transforms.json ─────────")
    transforms_file = output_path / "transforms.json"
    with open(transforms_file, "r") as f:
        data = json.load(f)

    # ns-process-data renames frames to frame_XXXXX.jpg (sorted by original name).
    # We need to map from the new sequentially-numbered names back to the original
    # vid_I_NNNN.jpg names so we can recover the source video index and frame number.
    orig_names_sorted = sorted(f.name for f in extracted_path.glob("vid_*.jpg"))

    frame_name_to_info: dict[str, dict] = {}
    for idx, orig_name in enumerate(orig_names_sorted):
        # ns-process-data produces frame_XXXXX.jpg starting from 00001.
        processed_frame_name = f"frame_{idx + 1:05d}.jpg"

        # Original name format: vid_<video_idx>_<frame_num>.jpg
        parts     = orig_name.split("_")        # ["vid", "<i>", "<NNNN>.jpg"]
        vid_idx   = int(parts[1])
        frame_num = int(parts[2].split(".")[0]) # 1-indexed as output by ffmpeg

        # ffmpeg's %04d counter is 1-indexed, so frame_num=1 → offset 0 seconds.
        local_time_sec   = (frame_num - 1) / fps
        abs_time_sec     = video_configs[vid_idx]["start_sec"] + local_time_sec
        normalized_time  = (abs_time_sec - global_start) / global_duration

        frame_name_to_info[processed_frame_name] = {
            "time":     round(float(normalized_time), 4),
            "video_id": vid_idx,
        }

    matched = 0
    for frame in data["frames"]:
        file_name = Path(frame["file_path"]).name
        if file_name in frame_name_to_info:
            info = frame_name_to_info[file_name]
            frame["time"]     = info["time"]
            frame["video_id"] = info["video_id"]
            matched += 1
        else:
            print(f"  ⚠️  Could not find temporal info for: {file_name}")

    print(f"  Injected temporal data into {matched:,}/{len(data['frames']):,} frames.")

    with open(transforms_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\n✅ Data preparation complete → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Section 5 — FULL PIPELINE
# ---------------------------------------------------------------------------

def videos_to_splat(
    meta_data: list[dict],
    output_dir: str = "./",
    fps: int = 5,
    iterations: int = 30000,
) -> Tuple[Path, SplatfactoModel]:
    """
    Full end-to-end pipeline: video file(s) → prepared dataset → trained splat model.

    This is the primary entry point when you want to go from raw video to a trained
    Gaussian Splat in a single function call. It chains:
      1. `prepare_datadir`    — frame extraction + COLMAP + temporal metadata
      2. `run_gs_reconstruction` — Splatfacto training

    Args:
        meta_data:   List of video config dicts (same format as `prepare_datadir`).
                     Each dict must have: "path", "start_sec", "end_sec".
        output_dir:  Root directory for all outputs.  The final layout is:
                       <output_dir>/data_prepared/   — nerfstudio dataset
                       <output_dir>/splatfacto/...   — checkpoints + config.yml
        fps:         Frames per second to extract from each video clip.
                     5 fps provides ~25% more COLMAP points than 4 fps for
                     better pose estimation and initial Gaussian placement.
        iterations:  Total Splatfacto training iterations.

    Returns:
        Tuple of:
          - config_yaml (Path): Path to config.yml for later reloading.
          - model (SplatfactoModel): Trained model in eval mode.
    """
    data_dir = prepare_datadir(
        meta_data,
        output_dir=os.path.join(output_dir, "data_prepared"),
        fps=fps,
    )
    config_path, model = run_gs_reconstruction(
        data_dir,
        output_dir=output_dir,
        iterations=iterations,
    )
    return config_path, model


# ---------------------------------------------------------------------------
# Section 6 — SPLAT DATA EXTRACTION
# ---------------------------------------------------------------------------

def get_splat_data(model: SplatfactoModel) -> dict[str, torch.Tensor]:
    """
    Extract raw Gaussian parameters from a trained model as plain PyTorch tensors.

    Useful when you want programmatic access to the splat data without writing
    to disk — e.g. to stream parameters to a renderer, serialise to a different
    format, or perform statistical analysis on the reconstructed scene.

    Args:
        model: A trained SplatfactoModel (from `run_gs_reconstruction` or
               `load_model_into_memory`).

    Returns:
        Dict with the following keys (all tensors on CPU, detached from autograd):
          - "xyz"       : [N, 3]  Gaussian centre positions (world space).
          - "colors"    : [N, 3]  Approximate RGB colours in [0, 1] (DC SH band only).
          - "opacities" : [N, 1]  Raw opacity logits (apply sigmoid for true opacity).
          - "scales"    : [N, 3]  Raw log-scales (apply exp for true scale in metres).
          - "quats"     : [N, 4]  Unit quaternions (w, x, y, z) defining orientation.
    """
    model.eval()
    with torch.no_grad():
        return {
            "xyz":       model.means.detach().cpu(),
            "colors":    model.colors.detach().cpu(),      # RGB ∈ [0, 1] via SH DC
            "opacities": model.opacities.detach().cpu(),   # raw logits; sigmoid → real α
            "scales":    model.scales.detach().cpu(),       # raw log-scales; exp → metres
            "quats":     model.quats.detach().cpu(),        # (w, x, y, z) unit quaternion
        }


# ---------------------------------------------------------------------------
# Section 7 — PLY EXPORT  (compatible with all standard GS viewers)
# ---------------------------------------------------------------------------

def export_splat_to_ply(
    model: SplatfactoModel,
    output_path: str | Path = "./splat.ply",
    color_mode: str = "rgb",
) -> None:
    """
    Export a trained Splatfacto model to a standard Gaussian Splatting .ply file.

    The output format matches the inria 3DGS convention and is compatible with:
      - antimatter15 / spz web viewer
      - supersplat editor
      - PlayCanvas engine
      - luma AI viewer
      - any viewer that supports the standard 3DGS PLY spec

    Two colour modes are supported:
      - "rgb"      : All colour information baked into uint8 R/G/B channels.
                     Compatible with *every* GS viewer. Recommended for sharing.
      - "sh_coeffs": Full spherical harmonics stored as f_dc_* and f_rest_* floats.
                     Required for view-dependent effects. Only supported in
                     SH-aware viewers (inria reference viewer, luma, etc.).

    Args:
        model:       A trained SplatfactoModel (in eval mode or training mode —
                     we call model.eval() internally).
        output_path: Destination .ply file path.
        color_mode:  "rgb" (default) or "sh_coeffs".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    map_to_tensors: OrderedDict[str, np.ndarray] = OrderedDict()

    with torch.no_grad():
        positions = model.means.cpu().numpy()   # [N, 3]
        n = positions.shape[0]

        # --- Positions ----------------------------------------------------------
        map_to_tensors["x"]  = positions[:, 0].astype(np.float32)
        map_to_tensors["y"]  = positions[:, 1].astype(np.float32)
        map_to_tensors["z"]  = positions[:, 2].astype(np.float32)

        # --- Normals (zeros) ----------------------------------------------------
        # Standard GS PLY must include nx/ny/nz fields even though they are unused
        # in rasterisation-based Gaussian Splatting renderers.
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        # --- Colours ------------------------------------------------------------
        if color_mode == "rgb":
            # Bake view-independent colour by evaluating SH at the DC band only.
            # Result is clamped and quantised to uint8 for maximum viewer compatibility.
            colors = torch.clamp(model.colors, 0.0, 1.0).cpu().numpy()
            colors_u8 = (colors * 255).astype(np.uint8)
            map_to_tensors["red"]   = colors_u8[:, 0]
            map_to_tensors["green"] = colors_u8[:, 1]
            map_to_tensors["blue"]  = colors_u8[:, 2]
        else:
            # SH DC band: shs_0 shape may be [N, 1, 3] or [N, 3] depending on
            # the nerfstudio version.  Handle both.
            shs_0 = model.shs_0.contiguous().cpu().numpy()
            if shs_0.ndim == 3:
                shs_0 = shs_0[:, 0, :]  # [N, 1, 3] → [N, 3]
            for i in range(3):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i].astype(np.float32)

            # SH higher-order bands.
            if model.config.sh_degree > 0:
                # shs_rest: [N, num_coeffs, 3] in nerfstudio.
                # Inria format expects: f_rest_0 … f_rest_{K-1} where the K coefficients
                # are stored channel-major (all R coefficients, then all G, then all B).
                # → transpose to [N, 3, num_coeffs], then flatten to [N, K*3].
                shs_rest = model.shs_rest.contiguous().cpu().numpy()    # [N, C, 3]
                shs_rest_t = shs_rest.transpose(0, 2, 1).reshape(n, -1)  # [N, C*3]
                for i in range(shs_rest_t.shape[1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest_t[:, i].astype(np.float32)

        # --- Opacity (raw logit) ------------------------------------------------
        # Stored as logit (pre-sigmoid) — this is the standard for 3DGS PLY so that
        # viewers can directly load and apply sigmoid during rendering.
        map_to_tensors["opacity"] = (
            model.opacities.data.cpu().numpy().squeeze().astype(np.float32)
        )

        # --- Scales (raw log-scale) ---------------------------------------------
        # Stored as log-scale; viewers/renderers apply exp() to get true scale.
        scales = model.scales.data.cpu().numpy()   # [N, 3]
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i].astype(np.float32)

        # --- Quaternions --------------------------------------------------------
        # Stored as (w, x, y, z) world-space unit quaternions.
        quats = model.quats.data.cpu().numpy()     # [N, 4]
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i].astype(np.float32)

    # --- Filter degenerate Gaussians --------------------------------------------
    # Three-pass filtering at export time to produce clean PLY files:
    #   1. NaN/Inf removal
    #   2. Low-opacity culling (sigmoid(logit) < 0.1 → nearly invisible)
    #   3. Oversized Gaussian removal (exp(log_scale) > scene_extent * 0.5)
    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        arr = t.reshape(n) if t.ndim > 1 else t
        select &= np.isfinite(arr)

    # Opacity: remove Gaussians with sigmoid(opacity) < 0.1
    # logit(0.1) ≈ -2.197. This is much more aggressive than the old logit(1/255)
    # threshold and catches the semi-transparent haze that makes splats look noisy.
    opa = map_to_tensors["opacity"]
    select &= opa >= -2.197

    # Scale: remove oversized floaters. Any Gaussian with exp(log_scale) > 0.5
    # in any axis is almost certainly a floater artifact, not real geometry.
    if all(f"scale_{i}" in map_to_tensors for i in range(3)):
        scale_arr = np.stack(
            [map_to_tensors[f"scale_{i}"] for i in range(3)], axis=-1
        )
        max_scale = np.exp(scale_arr.astype(np.float64)).max(axis=-1)
        select &= max_scale <= 0.5

    filtered_count = n - int(select.sum())
    if filtered_count > 0:
        print(f"  Filtered {filtered_count:,}/{n:,} Gaussians "
              f"(NaN / low-opacity / oversized)")
        for k in map_to_tensors:
            map_to_tensors[k] = map_to_tensors[k][select]
        n = int(select.sum())

    # --- Write binary PLY -------------------------------------------------------
    _write_gs_ply(str(output_path), n, map_to_tensors)
    print(f"✅ Exported {n:,} Gaussians → {output_path}")


def _write_gs_ply(
    filename: str,
    count: int,
    map_to_tensors: OrderedDict[str, np.ndarray],
) -> None:
    """
    Write a binary-little-endian PLY file in standard Gaussian Splatting format.

    This implementation uses a single structured-array binary dump rather than
    a row-by-row Python loop, making it orders of magnitude faster for large
    splats (e.g. 100k+ Gaussians write in milliseconds instead of tens of seconds).

    The PLY property type mapping follows the inria 3DGS convention:
      - float32 arrays → "property float <name>"
      - uint8 arrays   → "property uchar <name>"

    Args:
        filename:       Destination file path (will be overwritten).
        count:          Number of Gaussians (vertices) to write.
        map_to_tensors: Ordered dict mapping property name → 1-D numpy array of length
                        `count`. All arrays must already be filtered/sliced to length
                        `count` before calling this function.
    """
    with open(filename, "wb") as f:
        # --- PLY header ---------------------------------------------------------
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"comment Generated by Nerf4Dgsplat\n")
        f.write(f"element vertex {count}\n".encode())

        for key, tensor in map_to_tensors.items():
            if tensor.dtype.kind == "f":
                dtype_str = "float"
            elif tensor.dtype == np.uint8:
                dtype_str = "uchar"
            else:
                raise ValueError(
                    f"Unsupported dtype {tensor.dtype!r} for PLY property '{key}'. "
                    "Only float32 and uint8 are supported by the GS PLY spec."
                )
            f.write(f"property {dtype_str} {key}\n".encode())

        f.write(b"end_header\n")

        # --- Binary data — single structured-array dump -------------------------
        # Build a numpy structured dtype matching the PLY header order, then fill
        # it column-by-column and dump the entire buffer at once. This is ~100–1000×
        # faster than the equivalent row-by-row Python loop for large splats.
        dtype_list = [
            (key, arr.dtype) for key, arr in map_to_tensors.items()
        ]
        structured = np.empty(count, dtype=dtype_list)
        for key, arr in map_to_tensors.items():
            structured[key] = arr

        f.write(structured.tobytes())


# ---------------------------------------------------------------------------
# Section 8 — CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line interface for the Nerf4Dgsplat pipeline.

    Supports two modes:
      1. Full pipeline (default): video → data prep → training → PLY export
      2. Export-only (--export_only): re-export a PLY from an existing checkpoint

    Examples:
        # Full pipeline with a single video clip
        python Nerf4Dgsplat.py \\
            --video ./IMG_0040.MOV \\
            --start_sec 0 --end_sec 8 \\
            --fps 5 --iterations 30000 \\
            --output_dir ./results

        # Re-export checkpoint to PLY (no retraining)
        python Nerf4Dgsplat.py \\
            --export_only \\
            --ckpt_path ./results/splatfacto/<timestamp>/step-030000.ckpt \\
            --output_dir ./results
    """
    parser = argparse.ArgumentParser(
        description="Nerf4Dgsplat — Video-to-Gaussian-Splat Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Video input (full pipeline only) --------------------------------------
    parser.add_argument(
        "--video", type=str,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--video_url", type=str,
        help="Direct URL to a video file for downloading (e.g., from Filebin).",
    )
    parser.add_argument(
        "--start_sec", type=float, default=0.0,
        help="Start time within the video (seconds).",
    )
    parser.add_argument(
        "--end_sec", type=float, default=None,
        help="End time within the video (seconds). Defaults to the full video duration.",
    )

    # --- Training options -------------------------------------------------------
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Frames per second to extract from the video for COLMAP processing.",
    )
    parser.add_argument(
        "--iterations", type=int, default=30000,
        help="Number of Splatfacto training iterations (30K recommended, 10K for previews).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Root output directory for all results (dataset, checkpoints, PLY files).",
    )
    parser.add_argument(
        "--color_mode", type=str, default="rgb", choices=["rgb", "sh_coeffs"],
        help=(
            '"rgb": export baked RGB colours (works in all viewers). '
            '"sh_coeffs": export full SH data (view-dependent, advanced viewers only).'
        ),
    )

    # --- Export-only mode -------------------------------------------------------
    parser.add_argument(
        "--export_only", action="store_true",
        help="Skip training; re-export PLY files from an existing checkpoint.",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help=(
            "Path to a step-XXXXXX.ckpt file. Required when --export_only is set. "
            "Produces both an RGB PLY and an SH PLY in --output_dir."
        ),
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Export-only mode -------------------------------------------------------
    if args.export_only:
        if not args.ckpt_path:
            parser.error("--export_only requires --ckpt_path.")
        export_from_checkpoint(
            ckpt_path=args.ckpt_path,
            out_rgb=output_path / "splat_rgb.ply",
            out_sh=output_path  / "splat_sh.ply",
        )
        print(f"\n🎉 Export complete!")
        print(f"   RGB PLY: {output_path / 'splat_rgb.ply'}")
        print(f"   SH  PLY: {output_path / 'splat_sh.ply'}")
        return

    # --- Full pipeline ----------------------------------------------------------
    temp_video_path = None
    if args.video_url:
        print(f"\n─── Starting Direct Video Download ────────────────────────────────────")
        print(f"  URL: {args.video_url}")
        try:
            temp_video_path = f"temp_{uuid.uuid4()}.mp4"
            with requests.get(args.video_url, stream=True) as r:
                r.raise_for_status()
                with open(temp_video_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            args.video = temp_video_path
            print(f"  ✅ Download complete → {temp_video_path}")
        except Exception as e:
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            parser.error(f"Failed to download video from URL: {e}")

    if not args.video:
        parser.error("--video or --video_url is required unless --export_only is set.")

    try:
        # Determine video end time if not provided
        if args.end_sec is None:
            try:
                import cv2
                cap = cv2.VideoCapture(args.video)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_fps    = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                args.end_sec = total_frames / video_fps if video_fps > 0 else 30.0
                print(f"  Detected video duration: {args.end_sec:.1f}s")
            except Exception:
                args.end_sec = 30.0
                print("  ⚠️  Could not detect video duration; defaulting to 30s.")

        video_configs = [{
            "path":      args.video,
            "start_sec": args.start_sec,
            "end_sec":   args.end_sec,
        }]

        config_path, model = videos_to_splat(
            video_configs,
            output_dir=str(output_path),
            fps=args.fps,
            iterations=args.iterations,
        )

        # Export PLY results
        baked_path    = output_path / "baked_splat.ply"
        baked_sh_path = output_path / "baked_splat_sh.ply"

        print(f"\n─── Exporting PLY files ────────────────────────────────────────────────")
        export_splat_to_ply(model, baked_path,    color_mode="rgb")
        export_splat_to_ply(model, baked_sh_path, color_mode="sh_coeffs")

        print(f"\n🎉 Pipeline complete!")
        print(f"   Config:       {config_path}")
        print(f"   PLY (RGB):    {baked_path}   ← use this in most viewers")
        print(f"   PLY (SH):     {baked_sh_path}  ← higher quality in advanced viewers")
        print(f"\nTo reload the model later:")
        print(f'   pipeline, model = load_model_into_memory("{config_path}")')


    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"  🧹 Cleaned up temporary video: {temp_video_path}")


if __name__ == "__main__":
    main()


# a, model = videos_to_splat(
#         meta_data=[{
#             "path": "./Video Project 1.mp4",
#             "start_sec": 10,
#             "end_sec": 100,
#         }, {
#             "path": "./Video Project 2.mp4",
#             "start_sec": 10,
#             "end_sec": 100,
#         }],
#         output_dir="./results",
#         fps=5,
#         iterations=30000,
#     )
#     export_splat_to_ply(model)