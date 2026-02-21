import io
import os
import numpy as np
import torch

def convert_ply_to_splat(ply_path: str):
    if not os.path.exists(ply_path):
        print(f"Error: {ply_path} not found")
        return

    print(f"Parsing {ply_path}...")
    with open(ply_path, 'rb') as f:
        header = ""
        while "end_header" not in header:
            line = f.readline().decode('ascii', errors='ignore')
            if not line: break
            header += line
        
        data_start = f.tell()
        print(f"Header end at offset: {data_start}")
    
    num_vertices = 0
    for line in header.split('\n'):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
            break
            
    print(f"Num vertices: {num_vertices}")
    if num_vertices == 0:
        return

    # Standard Nerf4Dgsplat layout: 
    # floats: x,y,z, nx,ny,nz, uchar: r,g,b, float: opacity, scale[3], rot[4]
    dt = np.dtype([
        ('pos', 'f4', 3),
        ('normals', 'f4', 3),
        ('color', 'u1', 3),
        ('opacity', 'f4'),
        ('scale', 'f4', 3),
        ('rot', 'f4', 4)
    ])
    
    try:
        data = np.fromfile(ply_path, dtype=dt, count=num_vertices, offset=data_start)
        print(f"Successfully read {len(data)} vertices from file.")
    except Exception as e:
        print(f"Error reading with numpy: {e}")
        return

    if len(data) < num_vertices:
        print(f"Warning: Only read {len(data)} vertices, expected {num_vertices}")

    # Pack into .splat format
    rot = data['rot']
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot_unit = rot / (rot_norm + 1e-8)
    rot_u8 = ((rot_unit + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    
    opacity = np.clip(data['opacity'], 0, 1) * 255
    rgba_u8 = np.concatenate([
        data['color'], 
        opacity.reshape(-1, 1).astype(np.uint8)
    ], axis=1)
    
    buffer = io.BytesIO()
    for i in range(len(data)):
        buffer.write(data['pos'][i].tobytes())
        buffer.write(data['scale'][i].tobytes())
        buffer.write(rgba_u8[i].tobytes())
        buffer.write(rot_u8[i].tobytes())
        
    res = buffer.getvalue()
    print(f"Generated buffer size: {len(res)} bytes")
    print(f"Multiple of 4? {len(res) % 4 == 0}")
    print(f"Expected size: {len(data) * 32}")

if __name__ == "__main__":
    convert_ply_to_splat("baked_splat.ply")
