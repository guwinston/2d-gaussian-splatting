import os
import sys
import json
import torch
import numpy as np
import torchvision
from pathlib import Path
from types import SimpleNamespace
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from utils.graphics_utils import focal2fov, fov2focal
from utils.camera_utils import Camera
from scene.dataset_readers import CameraInfo

import einops
from copy import deepcopy
from e3nn import o3
from einops import einsum
from scipy.spatial.transform import Rotation
from utils.general_utils import build_rotation



@torch.no_grad()
def visualize_camera_and_gaussians(cameras, model):
    import open3d as o3d
    from tqdm import tqdm
    points = model.get_xyz.detach().cpu().numpy()
    colors = model._features_dc.detach().cpu().numpy().reshape(points.shape[0], -1)
    colors = colors * 0.28209479177387814 + 0.5
    print("model has {} gaussians".format(len(points)))
    print("points min {}, max {}, center {}".format(points.min(0), points.max(0), points.mean(0)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    base_point = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
    geometries = [pcd, base_point]

    for camera in tqdm(cameras):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=camera.image_width,
            height=camera.image_height,
            fx=fov2focal(camera.FoVx, camera.image_width),
            fy=fov2focal(camera.FoVy, camera.image_height),
            cx=camera.image_width/2,
            cy=camera.image_height/2
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

    o3d.visualization.draw_geometries(geometries, point_show_normal=False, height=1080, width=1920)

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


@torch.no_grad()
def rotate_xyz(gaussians, rotmat):
    new_xyz = gaussians.get_xyz
    mean_xyz = torch.mean(new_xyz,0)
    new_xyz = new_xyz - mean_xyz
    new_xyz = new_xyz @ rotmat.T
    gaussians._xyz = new_xyz + mean_xyz

@torch.no_grad()
def rotate_rot(gaussians, rotmat):
    new_rotation = build_rotation(gaussians._rotation)
    new_rotation = rotmat @ new_rotation
    new_quat = np.array(Rotation.from_matrix(new_rotation.detach().cpu().numpy()).as_quat())
    new_quat[:, [0,1,2,3]] = new_quat[:, [3,0,1,2]] # xyzw -> wxyz
    gaussians._rotation = torch.from_numpy(new_quat).to(gaussians._rotation.device).float()


@torch.no_grad()
def rotate_shs(gaussians, rotmat):
    # reference: https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
    shs_feat = gaussians._features_rest
    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32) # switch axes: yzx -> xyz
    permuted_rotmat = np.linalg.inv(P) @ rotmat.to('cpu').numpy() @ P
    rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotmat))

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(shs_feat.device).float()
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(shs_feat.device).float()
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(shs_feat.device).float()

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D_3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    gaussians._features_rest = shs_feat.float()

def rotate_camera(cam_position, cam_rotation, obj_position, rotmat):
    translated_position = cam_position - obj_position
    translated_position = translated_position @ rotmat.T
    new_cam_position = translated_position + obj_position
    new_cam_rotation = rotmat @ cam_rotation
    return new_cam_position, new_cam_rotation


def rotate_o3dmesh(mesh_path, mesh_save_path, rotation_matrix, model_center):
    import open3d as o3d
    mesh_dir = os.path.dirname(mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3."
    vertices = np.asarray(mesh.vertices)
    vertices -= model_center
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    rotated_vertices += model_center
    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)


def swap_xy_shs_vectorized(gaussians):
    """
    将右手坐标系转换为左手坐标系，通过交换 x 和 y 轴。
    假设 gaussians._features_rest 的形状为 [N, 3, 16]，
    各阶分别为 l=0:1 个, l=1:3 个, l=2:5 个, l=3:7 个系数。
    """
    shs_feat = gaussians._features_rest.clone().float()  # [N, 15, 3]
    shs_feat = shs_feat.permute(0, 2, 1)  # [N, 3, 15]

    # 定义每个阶的起始和结束索引
    degree_offsets = {
        1: (0, 3),
        2: (3, 8),
        3: (8, 15)
    }

    # 处理 l=1
    start, end = degree_offsets[1]
    # 假设 l=1 的系数顺序为 [Y_{1,-1}, Y_{1,0}, Y_{1,1}]
    # 交换 x 和 y 轴对应了 Y_{1,1} <-> Y_{1,-1}，可能需要改变符号
    l1 = shs_feat[:, :, start:end].clone()  # [N, 3, 3]
    shs_feat[:, :, start:end] = torch.stack([
        l1[:, :, 2],  # Y_{1,1} -> Y_{1,-1}
        l1[:, :, 1],  # Y_{1,0} 保持不变
        l1[:, :, 0],  # Y_{1,-1} -> Y_{1,1}
    ], dim=2)

    # 处理 l=2
    start, end = degree_offsets[2]
    # 假设 l=2 的系数顺序为 [Y_{2,-2}, Y_{2,-1}, Y_{2,0}, Y_{2,1}, Y_{2,2}]
    l2 = shs_feat[:, :, start:end].clone()  # [N, 3, 5]
    print(l2.shape)
    shs_feat[:, :, start:end] = torch.stack([
        l2[:, :, 4],  # Y_{2,2} -> Y_{2,-2}
        l2[:, :, 3],  # Y_{2,1} -> Y_{2,-1}
        l2[:, :, 2],  # Y_{2,0} 保持不变
        l2[:, :, 1],  # Y_{2,-1} -> Y_{2,1}
        l2[:, :, 0],  # Y_{2,-2} -> Y_{2,2}
    ], dim=2)

    # 处理 l=3
    start, end = degree_offsets[3]
    # 假设 l=3 的系数顺序为 [Y_{3,-3}, Y_{3,-2}, Y_{3,-1}, Y_{3,0}, Y_{3,1}, Y_{3,2}, Y_{3,3}]
    l3 = shs_feat[:, :, start:end].clone()  # [N, 3, 7]
    shs_feat[:, :, start:end] = torch.stack([
        l3[:, :, 6],  # Y_{3,3} -> Y_{3,-3}
        l3[:, :, 5],  # Y_{3,2} -> Y_{3,-2}
        l3[:, :, 4],  # Y_{3,1} -> Y_{3,-1}
        l3[:, :, 3],  # Y_{3,0} 保持不变
        l3[:, :, 2],  # Y_{3,-1} -> Y_{3,1}
        l3[:, :, 1],  # Y_{3,-2} -> Y_{3,2}
        l3[:, :, 0],  # Y_{3,-3} -> Y_{3,3}
    ], dim=2)

    gaussians._features_rest = shs_feat.permute(0, 2, 1)

    return gaussians

def rotate_and_swap_shs(gaussians, rotmat):
    gaussians = swap_xy_shs_vectorized(gaussians)
    rotate_shs(gaussians, rotmat)
    return gaussians

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

def convert_colmap_to_ue_camera(colmap_camera):
    w2c = np.eye(4)
    w2c[:3, :3] = colmap_camera.R.transpose()
    w2c[:3, 3] = colmap_camera.T
    c2w = np.linalg.inv(w2c)
    R, T = c2w[:3, :3], c2w[:3, 3]
    M = np.array([[0, 1, 0],[1, 0, 0],[0, 0, 1]])
    R = M @ R @ M.T
    T = M @ T

    roty_90 = get_axis_angle_rotation(R[:, 1], np.deg2rad(90)) 
    R = roty_90 @ R

    ue_c2w = np.eye(4)
    ue_c2w[:3, :3] = R
    ue_c2w[:3, 3] = T
    ue_w2c = np.linalg.inv(ue_c2w)
    ue_camera = CameraInfo(uid=colmap_camera.uid, R=ue_c2w[:3, :3], T=ue_w2c[:3, 3], FovY=colmap_camera.FovY, FovX=colmap_camera.FovX,
                           image=colmap_camera.image, image_path=colmap_camera.image_path, image_name=colmap_camera.image_name,
                           width=colmap_camera.width, height=colmap_camera.height)
    return ue_camera

@torch.no_grad()
def convert_colmap_to_ue_model(gaussians):
    device = gaussians._xyz.device
    M = np.array([[0, 1, 0],[1, 0, 0],[0, 0, 1]])
    M = torch.from_numpy(M).to(device).float()
    ue_model = deepcopy(gaussians)
    ue_model._xyz[:, [0, 1]] = ue_model._xyz[:, [1, 0]]
    ue_model._scaling[:, [0, 1]] = ue_model._scaling[:, [1, 0]]
    ue_model._rotation = build_rotation(ue_model._rotation)
    ue_model._rotation = M @ ue_model._rotation @ M.T
    ue_model = swap_xy_shs_vectorized(ue_model)
    return ue_model


if __name__ == "__main__":
    camera_path = "output/garden/cameras.json"
    model_path = "output/garden/point_cloud/iteration_30000/point_cloud.ply"
    model = load_model(model_path, sh_degree=3)
    ue_model = convert_colmap_to_ue_model(model)
    cameras_infos = load_camera_infos(camera_path)
    colmap_camera = cameras_infos[0]
    ue_camera = convert_colmap_to_ue_camera(colmap_camera)
    camera = Camera(colmap_id=ue_camera.uid, R=ue_camera.R, T=ue_camera.T, 
                    FoVx=ue_camera.FovX, FoVy=ue_camera.FovY, 
                    image=torch.zeros((3, ue_camera.height, ue_camera.width)), gt_alpha_mask=None,
                    image_name=ue_camera.image_name, uid=ue_camera.uid)
    rendering_image, rendering_depth, rendering_normal = render_image(ue_model, camera, white_backgrund=False) # hwc
    # torchvision.utils.save_image(torch.tensor((rendering_normal+1)/2).permute(2,0,1), "output/garden_aligned_xy/rendering_normal.png")
    # torchvision.utils.save_image(torch.tensor(rendering_image).permute(2,0,1), "output/garden_aligned_xy/rendering.png")
    
    visualize_camera_and_gaussians([camera], ue_model)
    import matplotlib.pyplot as plt
    plt.imshow(rendering_image)
    plt.show()
    


