import numpy as np
import os

def inspect_ply(ply_path):
    with open(ply_path, 'rb') as f:
        header = ""
        while "end_header" not in header:
            header += f.readline().decode('ascii', errors='ignore')
        data_start = f.tell()

    dt = np.dtype([
        ('pos', 'f4', 3),
        ('normals', 'f4', 3),
        ('color', 'u1', 3),
        ('opacity', 'f4'),
        ('scale', 'f4', 3),
        ('rot', 'f4', 4)
    ])
    
    num_vertices = 0
    for line in header.split('\n'):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
            break
            
    print(f"Num vertices: {num_vertices}")
    data = np.fromfile(ply_path, dtype=dt, count=10, offset=data_start)
    
    print("\n--- Vertex Samples (First 3) ---")
    for i in range(3):
        print(f"Vertex {i}:")
        print(f"  Pos: {data['pos'][i]}")
        print(f"  Color: {data['color'][i]}")
        print(f"  Opacity: {data['opacity'][i]}")
        print(f"  Scale: {data['scale'][i]}")
        print(f"  Rot: {data['rot'][i]}")

    # Read more for stats
    data_stats = np.fromfile(ply_path, dtype=dt, count=1000, offset=data_start)
    print("\n--- Statistics (1000 samples) ---")
    print(f"Opacity range: {data_stats['opacity'].min():.4f} to {data_stats['opacity'].max():.4f}")
    print(f"Scale range: {data_stats['scale'].min():.4f} to {data_stats['scale'].max():.4f}")
    print(f"Color range (red): {data_stats['color'][:,0].min()} to {data_stats['color'][:,0].max()}")

if __name__ == "__main__":
    inspect_ply("baked_splat.ply")
