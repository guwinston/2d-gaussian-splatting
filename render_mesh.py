import cv2
import json
import math
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftPhongShader, 
    AmbientLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex

from utils.graphics_utils import focal2fov, fov2focal
from scene.dataset_readers import CameraInfo



def load_ply_with_colors(file_path, device):
    # Load the ply file
    ply_data = PlyData.read(file_path)

    # Extract vertex data
    vertex_data = ply_data['vertex'].data
    verts = torch.tensor(np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1), dtype=torch.float32, device=device)

    # Extract face data and convert to int64
    face_data = ply_data['face'].data['vertex_indices']
    faces = torch.tensor(np.stack(face_data).astype(np.int64), dtype=torch.int64, device=device)

    # Extract color data if available
    if 'red' in vertex_data.dtype.names:
        verts_rgb = torch.tensor(np.stack([vertex_data['red'], vertex_data['green'], vertex_data['blue']], axis=-1), dtype=torch.float32, device=device) / 255.0
    else:
        # Default to white if no color information is available
        verts_rgb = torch.ones_like(verts, dtype=torch.float32, device=device)

    return verts, faces, verts_rgb

def load_obj_with_colors(file_path, device):
    verts = []
    faces = []
    verts_rgb = []

    # Manually parse the OBJ file
    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines(), desc='Loading OBJ file'):
            if line.startswith('v '):  # Vertex with color
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                r, g, b = map(float, parts[4:7])  # Assuming colors are in [0, 1] range
                verts.append([x, y, z])
                verts_rgb.append([r, g, b])
            elif line.startswith('f '):  # Face
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # OBJ is 1-indexed
                faces.append(face)

    verts = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    verts_rgb = torch.tensor(verts_rgb, dtype=torch.float32, device=device)

    return verts, faces, verts_rgb

def load_mesh_from_file(file_path, device):
    if file_path.endswith('.ply'):
        verts, faces_idx, verts_rgb = load_ply_with_colors(file_path, device)
    elif file_path.endswith('.obj'):
        verts, faces_idx, verts_rgb = load_obj_with_colors(file_path, device)
    else:
        raise ValueError("Unsupported file format. Supported formats are .ply and .obj.")
    return verts, faces_idx, verts_rgb

@torch.no_grad()
def visualize_camera_and_gaussians(cameras, mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
    geometries = [mesh, world_frame]

    for camera in tqdm(cameras):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=camera.width,
            height=camera.height,
            fx=fov2focal(camera.FovX, camera.width),
            fy=fov2focal(camera.FovY, camera.height),
            cx=camera.width/2,
            cy=camera.height/2
        )
        
        position = np.array(camera.T)               # W2C
        rotation = np.array(camera.R).transpose()   # W2C
        W2C = np.eye(4)
        W2C[:3, :3] = rotation
        W2C[:3,  3] = position
        C2W = np.linalg.inv(W2C)

        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
        camera_frame.transform(C2W)
        
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic.width, intrinsic.height, intrinsic.intrinsic_matrix, W2C, scale=2.0
        )

        geometries.append(camera_frame)
        geometries.append(frustum)

    o3d.visualization.draw_geometries(geometries, point_show_normal=True, height=1080, width=1920)


def load_camera_infos(camera_path):
    with open(camera_path) as f:
        cameras_data = json.load(f)

    cameras = []
    for cam_idx in range(len(cameras_data)):
        rotation = np.array(cameras_data[cam_idx]['rotation'])
        position = np.array(cameras_data[cam_idx]['position'])
        height = cameras_data[cam_idx]['height']
        width  = cameras_data[cam_idx]['width']
        focalx = cameras_data[cam_idx]['fx']
        focaly = cameras_data[cam_idx]['fy']
        id     = cameras_data[cam_idx]['id']
        name   = cameras_data[cam_idx]['img_name']

        C2W = np.zeros((4,4))
        C2W[:3, :3] = rotation
        C2W[:3,  3] = position
        C2W[3,   3] = 1
        W2C = np.linalg.inv(C2W)
        T = W2C[:3, 3]
        R = W2C[:3, :3].transpose()
        
        fov_y = focal2fov(focaly, height)
        fov_x = focal2fov(focalx, width)
        image = None

        # CameraInfo的输入的R和T需要特别注意，R为camera to world，T为world to camera
        camera = CameraInfo(uid=cam_idx, R=R, T=T, FovY=fov_y, FovX=fov_x, image=image,
                              image_path=None, image_name=name, width=width, height=height)
        cameras.append(camera)
    print(f"{len(cameras)} cameras have been loaded from {camera_path}")
    return cameras

def convert_colmap_to_pytorch3d_camera(cam_info, device):
    w2c = np.eye(4)
    w2c[:3, :3] = cam_info.R.transpose()
    w2c[:3, 3] = cam_info.T
    w2c = torch.from_numpy(w2c).float()
    c2w = torch.inverse(w2c)
    R, T = c2w[:3, :3], c2w[:3, 3:]
    R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to LUF for Rotation

    new_c2w = torch.cat([R, T], 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
    R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix
    R = R[None] # batch 1 for rendering
    T = T[None] # batch 1 for rendering
    H, W = cam_info.height, cam_info.width
    fx = fov2focal(cam_info.FovX, W)
    fy = fov2focal(cam_info.FovY, H)
    cx = W / 2
    cy = H / 2

    image_size = ((H, W),)  # (h, w)
    fcl_screen = ((fx, fy), )  # fcl_ndc * min(image_size) / 2
    prp_screen = ((cx, cy), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
    cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size, R=R, T=T, device=device)
    return cameras

def get_axis_angle_rotation(axis, theta, left_hand=False):
    # 需要注意的是，罗德里格公式默认定义为绕右手坐标系中的axis旋转theta角度，得到位于右手坐标系中的旋转矩阵
    # 如果是绕左手坐标系中轴旋转theta角度，需要将旋转角度theta取负，才能得到在左手坐标系中对应的正确旋转矩阵
    if left_hand:
        theta = -theta
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)  # normalize axis
    x, y, z = axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta

    # Constructing a Rotation Matrix Using the Rodrigues Rotation Formula
    rotation = np.array([
        [cos_theta + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_theta, x * z * one_minus_cos + y * sin_theta],
        [y * x * one_minus_cos + z * sin_theta, cos_theta + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_theta],
        [z * x * one_minus_cos - y * sin_theta, z * y * one_minus_cos + x * sin_theta, cos_theta + z * z * one_minus_cos]
    ])
    return rotation

def convert_ue_to_colmap_camera(position, quanternion, height, width, fov):
    # # 方式1：通过旋转矩阵
    R = Rotation.from_quat(quanternion).as_matrix()
    T = position
    M = np.array([[0, 1, 0],[1, 0, 0],[0, 0, 1]])
    R = M @ R @ M.T
    T = M @ T

    # 方式2：通过交换四元数分量
    # new_quat = [quanternion[1], quanternion[0], quanternion[2], -quanternion[3]] # 手性变化，w分量需要变为负数（或者xyz取负）
    # new_pos  = [position[1], position[0], position[2]]
    # R = Rotation.from_quat(new_quat).as_matrix()
    # T = np.array(new_pos)

    # rotx_90 = Rotation.from_rotvec(np.deg2rad(-90) * np.array(R[:, 0]))
    # rotx_90 = get_axis_angle_rotation(R[:, 0], np.deg2rad(-90)) # 注意不是绕世界坐标系的X轴旋转，而是绕自身局部坐标系的X轴旋转
    # R = rotx_90 @ R
    # print(R, T)
    R = R @ Rotation.from_euler('x', -90, degrees=True).as_matrix() # 绕自身x旋转的另一种方法

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    w2c = np.linalg.inv(c2w)

    fovx = np.deg2rad(fov)
    fovy = 2 * np.arctan(np.tan(fovx / 2) * (height / width))

    colmap_camera = CameraInfo(uid=0, R=c2w[:3, :3], T=w2c[:3, 3], FovY=fovy, FovX=fovx,
                                image=None, image_path=None, image_name="",
                                width=width, height=height)
    return colmap_camera

def convert_colmap_to_ue_camera(colmap_camera):
    w2c = np.eye(4)
    w2c[:3, :3] = colmap_camera.R.transpose()
    w2c[:3, 3] = colmap_camera.T
    c2w = np.linalg.inv(w2c)
    R, T = c2w[:3, :3], c2w[:3, 3]
    M = np.array([[0, 1, 0],[1, 0, 0],[0, 0, 1]])

    # 方式1：正向思维，先进行手系转换，再旋转到目标位姿
    # （1）通过旋转矩阵
    R = M @ R @ M.T
    T = M @ T
    # # （2）通过交换四元数分量
    # new_quat = Rotation.from_matrix(R).as_quat()
    # new_quat = [new_quat[1], new_quat[0], new_quat[2], -new_quat[3]]
    # R = Rotation.from_quat(new_quat).as_matrix()
    # T = M @ T

    # roty_90 = get_axis_angle_rotation(R[:, 1], np.deg2rad(90), left_hand=True) # 注意为左手坐标系
    # R = roty_90 @ R
    R = R @ Rotation.from_euler('y', -90, degrees=True).as_matrix() # 绕自身x旋转的另一种方法

    # # 方式2：逆向思维，COLMAP坐标系中先转到UE坐标系相同位姿，再进行手系转换
    # roty_90 = get_axis_angle_rotation(R[:, 0], np.deg2rad(90)) 
    # R = roty_90 @ R
    # # R = R @ Rotation.from_euler('x', 90, degrees=True).as_matrix() # 绕自身x旋转的另一种方法
    # # （1）通过旋转矩阵
    # R = M @ R @ M.T # 手系转换
    # T = M @ T
    # # （2）通过交换四元数分量
    # # new_quat = Rotation.from_matrix(R).as_quat()
    # # new_quat = [new_quat[1], new_quat[0], new_quat[2], -new_quat[3]]
    # # R = Rotation.from_quat(new_quat).as_matrix()
    # # T = M @ T
    
    ue_rotation = R
    ue_position = T
    ue_quaternion = Rotation.from_matrix(ue_rotation).as_quat()
    print(ue_position, ue_quaternion)
    return ue_position, ue_quaternion


def get_renderer(image_size, cameras):
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    lights = AmbientLights(device=device)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(device=device, cameras=cameras,lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer

def render_mesh(mesh_path, colmap_camera, device):
    verts, faces, verts_rgb = load_mesh_from_file(mesh_path, device)
    verts_rgb = verts_rgb.unsqueeze(0)  # Now the shape is (1, V, C)
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    camera = convert_colmap_to_pytorch3d_camera(colmap_camera, device)
    renderer = get_renderer((colmap_camera.height, colmap_camera.width), camera)
    images = renderer(mesh)
    return images

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mesh_file = "output/lego/train/ours_30000/fuse_post.ply"
    camera_file = "output/lego/cameras.json"
    camera_infos = load_camera_infos(camera_file)
    # colmap_camera = camera_infos[0]
    # colmap_camera = CameraInfo(uid=camera_infos[0].uid, R=camera_infos[0].R, T=camera_infos[0].T, 
    #                            FovY=camera_infos[0].FovY, FovX=camera_infos[0].FovX, image=None, 
    #                            image_path=None, image_name=camera_infos[0].image_name, 
    #                            width=1920, height=1080)

    pos=[0.0674329376,-2.08051041,2.26654663]
    quat=[-0.283864349,0.249367356,0.695590079,0.61105758]
    colmap_camera = convert_ue_to_colmap_camera(position=np.array(pos), 
                                                quanternion=np.array(quat),
                                                height=1080, width=1920, fov=90)
    ue_position, ue_quaternion = convert_colmap_to_ue_camera(colmap_camera)
    print(f"original ue:\n    Position: {pos}\n    Quaternion: {quat}")
    print(f"ue covert to colmap:\n    Position: {(-colmap_camera.R @ colmap_camera.T).tolist()}\n    Quaternion: {Rotation.from_matrix(colmap_camera.R).as_quat().tolist()}")
    print(f"colmap covert to ue:\n    Position: {ue_position.tolist()}\n    Quaternion: {ue_quaternion.tolist()}")

    # # visualize_camera_and_gaussians([colmap_camera], mesh_file)
    # images = render_mesh(mesh_file, colmap_camera, device)
    # image_cv2 = cv2.cvtColor((images[0, ..., :3].cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    # cv2.imwrite("output/lego/mesh_rendering.png", image_cv2)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.show()
