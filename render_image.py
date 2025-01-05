import os
import sys
import cv2
import json
import torch
import numpy as np
import open3d as o3d
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from utils.graphics_utils import focal2fov, fov2focal
from utils.camera_utils import Camera
from scene.dataset_readers import CameraInfo

def load_model(model_path, sh_degree=3):
    with torch.no_grad():
        gaussians  = GaussianModel(sh_degree)
        gaussians.load_ply(model_path)
        print(f"gaussain model has been load from {model_path}")
        return gaussians

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

        # CameraInfo 和 Camera的输入的R和T需要特别注意，R为camera to world，T为world to camera
        # camera = Camera(colmap_id=id, R=R, T=T, FoVx=fov_x, FoVy=fov_y, image=image, gt_alpha_mask=None,
        #                 image_name=name, uid=id)
        camera = CameraInfo(uid=cam_idx, R=R, T=T, FovY=fov_y, FovX=fov_x, image=image,
                              image_path=None, image_name=name, width=width, height=height)
        cameras.append(camera)
    print(f"{len(cameras)} cameras have been loaded from {camera_path}")
    return cameras

def render_image(model, camera, image_save_path=None, white_backgrund=False):
    with torch.no_grad():
        pipeline   = {"debug":False, "compute_cov3D_python":False, "convert_SHs_python":False, "depth_ratio":0}
        pipeline   = SimpleNamespace(**pipeline)
        background = torch.tensor([0, 0, 0] if not white_backgrund else [1,1,1], dtype=torch.float32, device="cuda")
        render_pkg = render(camera, model, pipeline, background)
        rendering_image = render_pkg["render"]
        rendering_depth = render_pkg["surf_depth"]
        rendering_normal = render_pkg["rend_normal"]
        if image_save_path is not None:
            torchvision.utils.save_image(rendering_image, image_save_path)
    return rendering_image.permute(1,2,0).cpu().numpy(),  rendering_depth.permute(1,2,0).cpu().numpy(), rendering_normal.permute(1,2,0).cpu().numpy()


def get_axis_angle_rotation(axis, theta):
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
    # R = Rotation.from_quat(quanternion).as_matrix()
    # T = position
    # M = np.array([[0, 1, 0],[1, 0, 0],[0, 0, 1]])
    # R = M @ R @ M.T
    # T = M @ T

    # 方式2：通过交换四元数分量
    new_quat = [quanternion[1], quanternion[0], quanternion[2], -quanternion[3]] # 手性变化，w分量需要变为负数（或者xyz取负）
    new_pos  = [position[1], position[0], position[2]]
    R = Rotation.from_quat(new_quat).as_matrix()
    T = np.array(new_pos)

    # rotx_90 = Rotation.from_rotvec(np.deg2rad(-90) * np.array(R[:, 0]))
    rotx_90 = get_axis_angle_rotation(R[:, 0], np.deg2rad(-90)) # 注意不是绕世界坐标系的X轴旋转，而是绕自身局部坐标系的X轴旋转
    R = rotx_90 @ R
    print(R, T)

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

def create_ply_with_normals_and_colors(normal_map, depth_map, rgb_image, K, R, t, ply_filename, step=10):
    if t.shape == (3,):
        t = t[:, np.newaxis]
    H, W = depth_map.shape

    points = []
    normals = []
    colors = []
    for v in range(0, H, step):
        for u in range(0, W, step):
            d = depth_map[v, u]
            if d == 0:
                continue

            pixel_coord = np.array([u, v, 1])
            camera_coord = d * np.linalg.inv(K) @ pixel_coord
            world_coord = R @ camera_coord + t.flatten()

            normal = normal_map[v, u] / 255.0 * 2 - 1
            color = rgb_image[v, u] / 255.0

            points.append(world_coord)
            normals.append(normal)
            colors.append(color)

    points = np.array(points)
    normals = np.array(normals)
    colors = np.array(colors)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(ply_filename, point_cloud)
    print(f"Point cloud saved as {ply_filename}")
    o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)


if __name__ == "__main__":
    model = load_model(model_path="output/lego/point_cloud/iteration_30000/point_cloud.ply", sh_degree=3)
    camera_infos = load_camera_infos(camera_path="output/lego/cameras.json")

    # cam_info = camera_infos[0]
    # cam_info = CameraInfo(uid=camera_infos[0].uid, R=camera_infos[0].R, T=camera_infos[0].T, 
    #                         FovY=camera_infos[0].FovY, FovX=camera_infos[0].FovX, image=None, 
    #                         image_path=None, image_name=camera_infos[0].image_name, 
    #                         width=1920, height=1080)
    # cam_info = convert_ue_to_colmap_camera(position=np.array([-3.00,0,1.00]), 
    #                                         quanternion=np.array([0,0,0,1]),
    #                                         height=1080, width=1920, fov=90)
    pos=[0.0674329376,-2.08051041,2.26654663]
    quat=[-0.283864349,0.249367356,0.695590079,0.61105758]
    cam_info = convert_ue_to_colmap_camera(position=np.array(pos), 
                                            quanternion=np.array(quat),
                                            height=1080, width=1920, fov=90)
    camera = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=torch.zeros((3,cam_info.height, cam_info.width)), gt_alpha_mask=None,
                    image_name=cam_info.image_name, uid=cam_info.uid, 
                    )
    rendering_image, rendering_depth, rendering_normal = render_image(model, camera, white_backgrund=False) # hwc
    torchvision.utils.save_image(torch.tensor((rendering_normal+1)/2).permute(2,0,1), "output/lego/rendering_normal.png")
    torchvision.utils.save_image(torch.tensor(rendering_image).permute(2,0,1), "output/lego/rendering.png")
    plt.imshow(rendering_image)
    plt.show()

    # K = np.array([[fov2focal(camera.FoVx, camera.image_width), 0, camera.image_width / 2],
    #               [0, fov2focal(camera.FoVy, camera.image_height), camera.image_height / 2],
    #               [0, 0, 1]])
    # W2C = np.eye(4)
    # W2C[:3, :3] = cam_info.R.T
    # W2C[:3, 3] = cam_info.T
    # C2W = np.linalg.inv(W2C)
    # R = C2W[:3, :3]
    # t = C2W[:3, 3]
    # normal_map = ((rendering_normal+1)/2)*255  # (H, W, 3)
    # depth_map = rendering_depth[..., 0]   # (H, W)
    # rgb_map = rendering_image*255   # (H, W, 3)
    # create_ply_with_normals_and_colors(normal_map, depth_map, rgb_map, K, R, t, "output/garden/rendering.ply", step=10)