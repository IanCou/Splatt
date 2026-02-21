import torch
import torch.nn as nn
import os

class DeformationField(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)
        )

def generate_dummy_pt(output_path, num_gaussians=1000):
    # Create random data matching the Gaussian4DModel structure
    state_dict = {
        'xyz': torch.randn(num_gaussians, 3),
        'scales': torch.ones(num_gaussians, 3) * 0.1,
        'rotations': torch.randn(num_gaussians, 3),
        'colors': torch.rand(num_gaussians, 3),
        'opacities': torch.ones(num_gaussians) * 0.8,
        # Adding deformation field weights even if not used by renderer yet
        'deformation.net.0.weight': torch.randn(128, 4),
        'deformation.net.0.bias': torch.randn(128),
        'deformation.net.2.weight': torch.randn(128, 128),
        'deformation.net.2.bias': torch.randn(128),
        'deformation.net.4.weight': torch.randn(9, 128),
        'deformation.net.4.bias': torch.randn(9),
    }
    
    torch.save(state_dict, output_path)
    print(f"Generated dummy .pt file at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_dummy_pt("dummy_scene.pt")
