# videos_to_gsplat
# pip install tqdm
# pip install opencv-python
# pip install gsplat
# pip install torchgeometry

import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import logging
import time
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from gsplat.rendering import rasterization
from torchgeometry import angle_axis_to_quaternion

# gaussian_4d_model.pt

class GaussianModel:
    def __init__(self, num_gaussians):
        self.means = torch.randn(num_gaussians, 3)
        self.quats = torch.randn(num_gaussians, 4)
        self.scales = torch.randn(num_gaussians, 3)
        self.colors = torch.randn(num_gaussians, 3)
        self.opacities = torch.ones(num_gaussians)

class MultiVideoDataset(IterableDataset):
    def __init__(self, video_configs, global_start, global_end):
        """
        video_configs: list of dicts:
            {
                "path": "video.mp4",
                "start_time": float,
                "end_time": float,
                "intrinsics": tensor(3x3),
                "extrinsics": tensor(4x4)
            }
        """
        self.video_configs = video_configs
        self.global_start = global_start
        self.global_end = global_end

    def normalize_time(self, t):
        return (t - self.global_start) / (self.global_end - self.global_start)

    def __iter__(self):
        for config in self.video_configs:
            cap = cv2.VideoCapture(config["path"])
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = config["start_time"] + frame_idx / fps
                norm_time = self.normalize_time(timestamp)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).float() / 255.0

                yield {
                    "image": frame.permute(2, 0, 1),
                    "time": torch.tensor(norm_time).float(),
                    "intrinsics": config["intrinsics"],
                    "extrinsics": config["extrinsics"],
                }

                frame_idx += 1

            cap.release()

class DeformationField(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # xyz + t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)  # dx(3) + drot(3) + dscale(3)
        )

    def forward(self, xyz, t):
        t_expand = t.expand(xyz.shape[0], 1)
        inp = torch.cat([xyz, t_expand], dim=-1)
        out = self.net(inp)

        dx = out[:, :3]
        drot = out[:, 3:6]
        dscale = out[:, 6:9]

        return dx, drot, dscale

class Gaussian4DModel(nn.Module):
    def __init__(self, num_gaussians=50000):
        super().__init__()
        # Canonical Gaussians as learnable parameters
        self.xyz = nn.Parameter(torch.randn(num_gaussians, 3))       # positions
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3))     # size
        self.rotations = nn.Parameter(torch.zeros(num_gaussians, 3)) # rotation vector
        self.colors = nn.Parameter(torch.rand(num_gaussians, 3))     # RGB
        self.opacities = nn.Parameter(torch.ones(num_gaussians))     # alpha

        self.deformation = DeformationField()

    def forward(self, t):
        # Apply deformation
        dx, drot, dscale = self.deformation(self.xyz, t)
        xyz_def = self.xyz + dx
        scale_def = self.scales + dscale
        rot_def = self.rotations + drot
        return xyz_def, scale_def, rot_def

def train(
    model,
    dataloader,
    num_epochs=10,
    lr=1e-3,
    device="cuda",
    writer=None,
    checkpoint_dir=None,
    save_interval=5
):
    logging.info(f"Starting training for {num_epochs} epochs on {device}")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        num_batches = 0

        for batch in pbar:
            image = batch["image"].to(device)
            t = batch["time"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            extrinsics = batch["extrinsics"].to(device)

            xyz, scales, rotations = model(t)
            quat = angle_axis_to_quaternion(rotations)

            rendered, alphas, meta = rasterization(
                means=xyz,
                scales=scales,
                quats=quat,
                colors=model.colors,
                opacities=model.opacities,
                viewmats=extrinsics.unsqueeze(0),
                Ks=intrinsics.unsqueeze(0),
                height=image.shape[1],
                width=image.shape[2],
            )

            rendered = rendered.squeeze(0).permute(2, 0, 1)

            loss = ((rendered - image) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            if writer:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
                if device == "cuda":
                    writer.add_scalar("System/GPU_Mem_Allocated_GB", torch.cuda.memory_allocated() / 1e9, global_step)

            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.6f}")
            global_step += 1

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time
        logging.info(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.6f}. Total Elapsed: {elapsed:.2f}s")
        
        if checkpoint_dir and (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")

    logging.info("Training complete.")


def videos_to_gsplat(video_configs, pt_file_dir, log_dir="runs"):
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

    global_start = min(cfg["start_time"] for cfg in video_configs)
    global_end = max(cfg["end_time"] for cfg in video_configs)

    dataset = MultiVideoDataset(video_configs, global_start, global_end)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4)

    model = Gaussian4DModel(num_gaussians=500)  # smaller for testing
    
    writer = SummaryWriter(log_dir=log_dir)
    
    train(
        model, 
        dataloader, 
        num_epochs=2, 
        lr=1e-3, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        writer=writer,
        checkpoint_dir=log_dir
    )
    
    torch.save(model.state_dict(), pt_file_dir)
    writer.close()
    return model

def make_config(video_path, start, end):
    return         {
        "path": video_path,
        "start_time": start,
        "end_time": end,
        "intrinsics": torch.eye(3), #MUST BE 3
        "extrinsics": torch.eye(4) #MUST BE 4
        }


# if __name__ == "__main__":
#     configs = [
#         {
#         "path": "IMG_7434.MOV",
#         "start_time": 0.0,
#         "end_time": 11.0,
#         "intrinsics": torch.eye(3), #MUST BE 3
#         "extrinsics": torch.eye(4) #MUST BE 4
#         }]
#     videos_to_gsplat(configs, "TestSplat.pt")


x = [make_config("./Untitled.mp4", 0.0, 360.0)]
videos_to_gsplat(x, "./output.pt")

