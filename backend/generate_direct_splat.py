import numpy as np
import os

def generate_dummy_splat(output_path, num_gaussians=5000):
    # pos (float32, 3), scale (float32, 3), color (uint8, 4), rot (uint8, 4) = 32 bytes
    
    # Random positions (centered at origin)
    pos = np.random.randn(num_gaussians, 3).astype(np.float32)
    
    # Random small scales
    scale = np.random.uniform(0.01, 0.05, (num_gaussians, 3)).astype(np.float32)
    
    # Random colors (RGBA)
    color = np.random.randint(0, 256, (num_gaussians, 4), dtype=np.uint8)
    # Set alpha to something visible
    color[:, 3] = 200
    
    # Identity rotation (0,0,0,1) mapped to 0-255: (128, 128, 128, 255)
    rot = np.full((num_gaussians, 4), [128, 128, 128, 255], dtype=np.uint8)
    
    with open(output_path, 'wb') as f:
        for i in range(num_gaussians):
            f.write(pos[i].tobytes())
            f.write(scale[i].tobytes())
            f.write(color[i].tobytes())
            f.write(rot[i].tobytes())
            
    print(f"Directly generated .splat file at: {os.path.abspath(output_path)}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    generate_dummy_splat("dummy_scene.splat")
