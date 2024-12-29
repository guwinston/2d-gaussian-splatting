import open3d as o3d
import numpy as np
from tqdm import tqdm

def ply_to_obj(ply_file_path, obj_file_path):
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    o3d.io.write_triangle_mesh(obj_file_path, mesh)

def load_obj_file(obj_file_path):
    vertices = []
    colors = []
    faces = []
    normals = []
    with open(obj_file_path, 'r') as f:
        for line in tqdm(f.readlines(), desc='Loading OBJ file'):
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                if len(parts) > 4:
                    r, g, b = map(float, parts[4:7])
                    colors.append([r, g, b])
                else:
                    colors.append([0.5, 0.5, 0.5])
                vertices.append([x, y, z])
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
            elif line.startswith('vn '):
                parts = line.strip().split()
                normal = [float(p) for p in parts[1:]]
                normals.append(normal)

    return np.array(vertices), np.array(colors), np.array(faces), np.array(normals)


def write_obj_file(obj_save_path, vertices, colors, faces, normals):
    print("Writing {} vertices, {} faces, {} normals to {}".format(len(vertices), len(faces), len(normals), obj_save_path))
    assert len(vertices) == len(colors) and len(vertices) == len(normals)
    assert np.max(faces) < len(vertices) and np.min(faces) >= 0
    with open(obj_save_path, 'w') as f:
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        for face in faces:
            f.write(f"f {' '.join([f'{idx+1}//{idx+1}' for idx in face])}\n")


def exchange_xy_obj(file_path, obj_save_path):
    if file_path.endswith('.obj'):
        vertices, colors, faces, normals = load_obj_file(file_path)
    else:
        mesh = o3d.io.read_triangle_mesh(file_path)
        vertices = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        faces = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)
        if len(normals) == 0:
            print("No normals found, computing normals...")
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
    print("Loaded mesh with {} vertices and {} faces from {}".format(len(vertices), len(faces), file_path))
    
    vertices[:, [0, 1]] = vertices[:, [1, 0]]
    normals[:, [0, 1]] = normals[:, [1, 0]]
    faces[:, [0, 1, 2]] = faces[:, [2, 1, 0]] # back face通过顶点顺序来区分，对mesh进行镜像后，顶点顺序也需要交换才能正确定义back face
    
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    # o3d.io.write_triangle_mesh(obj_save_path, mesh)
    write_obj_file(obj_save_path, vertices, colors, faces, normals)
    print("Saved mesh to {}".format(obj_save_path))

def visualize_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    # mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name="Mesh Visualization")

# ply_file_path = "output/garden_aligned_xy/train/ours_30000/aligned_xy_mesh.ply"
# obj_file_path = "output/garden_aligned_xy/train/ours_30000/aligned_xy_mesh.obj"
# ply_to_obj(ply_file_path, obj_file_path)
# visualize_mesh(obj_file_path)


exchange_xy_obj("output/lego/train/ours_30000/fuse_post.ply", "output/lego/train/ours_30000/fuse_post.obj")
visualize_mesh("output/lego/train/ours_30000/fuse_post.obj")
