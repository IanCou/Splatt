#Usage: IDK I suffer u suffer
# pip install pyglomap
# pip install nerfstudio

from pathlib import Path
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.utils.eval_utils import eval_setup

import json
import subprocess
import os
from pathlib import Path
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def run_4d_gs_reconstruction(
    data_dir: str,
    video_meta: list[dict], # List of {'path': str, 'start_t': float, 'end_t': float}
    output_dir: str = "outputs/dynamic_gs",
    iterations: int = 5000
):
    """
    Runs a Dynamic Gaussian Splatting reconstruction optimized for 
    multiple videos with temporal offsets and potential occlusions.
    """
    
    config = TrainerConfig(
        method_name="splatfacto", # Using Splatfacto with a deformation extension
        max_num_iterations=iterations,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    data=Path(data_dir),
                    load_3D_points=True,
                    # We assume transforms.json includes 'time' and 'video_id'
                ),
                cache_images="gpu",
                cache_images_type="uint8"
            ),
            model=SplatfactoModelConfig(
                # 1. TEMPORAL REASONING: 
                # Note: Standard Splatfacto needs the 'dynamic' flag or a 
                # custom deformation extension to be truly 4D.
                # to optimize and speed up execution.
                stop_split_at=4000,          # Stop growing early
                cull_alpha_thresh=0.05,      # Aggressive cleaning
                densify_grad_thresh=0.005,   # Be picky about new points
                # 2. CAMERA OPTIMIZATION: Handles relative pose errors between videos
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", 
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-15)
                ),
                
                # 3. HANDLING BLUR/OCCLUSIONS:
                # We use 'use_appearance_embedding' to help the model 
                # 'ignore' transient objects like arms or motion blur artifacts.
                use_appearance_embedding=True,
                appearance_embed_dim=32,
                
                # Performance settings for 4D
                sh_degree=2,
                num_downscales=2,
            ),
        ),
        optimizers={
            "means": {"optimizer": AdamOptimizerConfig(lr=1.6e-4), "scheduler": None},
            "quats": {"optimizer": AdamOptimizerConfig(lr=1.0e-3), "scheduler": None},
            "scales": {"optimizer": AdamOptimizerConfig(lr=5e-3), "scheduler": None},
            "opacities": {"optimizer": AdamOptimizerConfig(lr=5e-2), "scheduler": None},
            "features_dc": {"optimizer": AdamOptimizerConfig(lr=2.5e-3), "scheduler": None},
            "features_rest": {"optimizer": AdamOptimizerConfig(lr=1.25e-4), "scheduler": None},
            # NEW: Optimizer for the Deformation Field (The 4th Dimension)
            "deformation_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=iterations),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        base_dir=Path(output_dir),
        mixed_precision=True
    )

    trainer = config.setup()
    trainer.train()
    return trainer.pipeline.model

def load_4d_model_into_memory(config_path: str):
    """
    Loads the YAML config and the associated model weights into a usable Pipeline.
    """
    config_path = Path(config_path)
    
    # eval_setup returns a tuple: (Config, Pipeline, CheckpointPath, Step)
    # The 'pipeline' object contains your model.
    config, pipeline, checkpoint_path, step = eval_setup(
        config_path,
        test_mode="inference" # "inference" disables training-specific gradients
    )
    
    print(f"Successfully loaded model from {checkpoint_path} at step {step}")
    
    # If you just want the 4D Model itself:
    model = pipeline.model
    model.eval() # Set to evaluation mode (crucial for 4D deformation)
    
    return pipeline, model

def prepare_4d_datadir(
    video_configs: list[dict], # list of {'path': 'v1.mp4', 'start_sec': 0, 'end_sec': 10}
    output_dir: str = "data_prepared",
    fps: int = 2
):
    """
    Extracts frames, runs COLMAP for camera poses, and injects 4D temporal data.
    """
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # 1. Determine Global Timeline
    all_starts = [v['start_sec'] for v in video_configs]
    all_ends = [v['end_sec'] for v in video_configs]
    global_start = min(all_starts)
    global_duration = max(all_ends) - global_start

    print(f"--- Extracting Frames and Calculating 4D Timestamps ---")
    
    for i, v in enumerate(video_configs):
        v_path = Path(v['path'])
        # Extract frames using ffmpeg
        # We name them video_{i}_frame_{n}.jpg to keep them unique
        cmd = [
            "ffmpeg", "-i", str(v_path), 
            "-vf", f"fps={fps}", 
            "-q:v", "2", 
            str(images_path / f"vid_{i}_%04d.jpg")
        ]
        subprocess.run(cmd, check=True)

    # 2. Run Nerfstudio Pose Estimation
    # This aligns all cameras from all videos into one 3D coordinate system.
    print(f"--- Running COLMAP (This may take a while) ---")
    subprocess.run([
        "ns-process-data", "images",
        "--data", str(images_path),
        "--output-dir", str(output_path),
        "--matcher-type", "sequential",
        "--num-frames-target", "100",
        "--max-num-features", "2000",
        "--sfm-tool", "glomap",
        "--no-verbose"
    ], check=True)

    # 3. Inject the "Time" Dimension into transforms.json
    transforms_file = output_path / "transforms.json"
    with open(transforms_file, "r") as f:
        data = json.load(f)

    print(f"--- Injecting Temporal Data into transforms.json ---")
    for frame in data["frames"]:
        # Extract video index and frame number from filename: vid_{i}_{num}.jpg
        file_name = Path(frame["file_path"]).name
        parts = file_name.split("_")
        vid_idx = int(parts[1])
        frame_num = int(parts[2].split(".")[0])

        # Calculate local time in seconds: (frame_num - 1) / fps
        local_time_sec = (frame_num - 1) / fps
        
        # Calculate global absolute time
        abs_time = video_configs[vid_idx]['start_sec'] + local_time_sec
        
        # Normalize to 0.0 - 1.0 for the 4D Gaussian model
        normalized_time = (abs_time - global_start) / global_duration
        
        # Add to the JSON object
        frame["time"] = round(float(normalized_time), 4)
        frame["video_id"] = vid_idx # Useful for appearance embeddings/blurry frames

    with open(transforms_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"--- Done! data_dir is ready at: {output_dir} ---")
    return str(output_path)

# Example Usage:
# videos = [
#     {"path": "drone_shot.mp4", "start_sec": 0, "end_sec": 30},
#     {"path": "handheld_close_up.mp4", "start_sec": 15, "end_sec": 45}
# ]
# prepare_4d_datadir(videos)


def videos_to_yml(meta_data, output_dir = "./", fps = 2):
    model = run_4d_gs_reconstruction(prepare_4d_datadir(meta_data, fps=fps), meta_data, output_dir=output_dir)
    return model

def get_3d_splat_at_time(model_path, global_time: float):
    """
    Extracts the deformed 3D Gaussian parameters at a specific timestamp.
    """
    model = load_4d_model_into_memory(model_path)[1]
    model.eval()
    
    # 1. Prepare the time input (must be a tensor on the correct device)
    # Most 4D models expect a tensor of shape [num_gaussians, 1]
    t = torch.full((model.num_gaussians, 1), global_time).to(model.device)
    
    with torch.no_grad():
        # 2. Query the deformation field
        # Note: Method names vary slightly by plugin, but 'get_gaussian_deltas' 
        # or 'deformation_field' are standard.
        offsets, scale_deltas, quat_deltas = model.deformation_field(model.means, t)
        
        # 3. Apply the deformation to the canonical (base) splat
        deformed_means = model.means + offsets
        deformed_scales = model.scales + scale_deltas
        deformed_quats = model.quats + quat_deltas
        
        # 4. Get the colors (usually static or SH-based)
        colors = model.get_colors() 

    return {
        "xyz": deformed_means,
        "rgba": torch.cat([colors, model.get_opacity()], dim=-1),
        "scales": deformed_scales,
        "quats": deformed_quats
    }

def export_4d_splat_to_ply(model, global_time: float, output_path = "./"):
    """
    Bakes the 4D deformation at a specific time and saves it as a 3D .ply file.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Calculate the deformation for the given time
    # We create a time tensor matching the number of gaussians
    t = torch.full((model.num_gaussians, 1), global_time).to(model.device)
    
    with torch.no_grad():
        # Get the offsets from the 4D deformation field
        # Note: In most 4D implementations, this returns (xyz_offsets, scale_offsets, quat_offsets)
        offsets, _, _ = model.deformation_field(model.means, t)
        
        # Apply offsets to get the 3D positions at this specific moment
        xyz = (model.means + offsets).cpu().numpy()
        
        # Get other attributes (colors, opacity, scales)
        # Colors are converted from Spherical Harmonics to basic RGB for standard PLY compatibility
        colors = model.get_colors().cpu().numpy() * 255
        opacities = model.get_opacity().cpu().numpy()
        scales = model.scales.cpu().numpy()
        quats = model.quats.cpu().numpy()

    # 2. Construct the PLY data structure
    # Standard Gaussian Splatting PLY format includes: x, y, z, nx, ny, nz, red, green, blue, opacity, etc.
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('opacity', 'f4')
    ]
    
    # We add scales and quats if you plan to use this in a specialized GS renderer
    for i in range(3): dtype.append((f'scale_{i}', 'f4'))
    for i in range(4): dtype.append((f'rot_{i}', 'f4'))

    elements = np.empty(xyz.shape[0], dtype=dtype)
    
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['red'], elements['green'], elements['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    elements['opacity'] = opacities.squeeze()
    
    for i in range(3): elements[f'scale_{i}'] = scales[:, i]
    for i in range(4): elements[f'rot_{i}'] = quats[:, i]

    # 3. Write to file
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(str(output_path))
    
    print(f"Exported baked 3D splat (time={global_time}) to {output_path}")

if __name__ == "__main__":
    pass
    # model = videos_to_yml([dict1, dict2...], fps=2) time is in seconds, fps is how many frames sampled per second
    # export_4d_splat_to_ply(model, time)
    # default put stuff in current directory
    # example {"path": "drone_shot.mp4", "start_sec": 0, "end_sec": 30},

