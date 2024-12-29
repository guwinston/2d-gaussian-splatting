#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import os
import torch
import torchvision
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from scene import Scene
from os import makedirs
from argparse import ArgumentParser
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from utils.render_utils import save_img_u8, save_img_f32

def mask_gaussians(gaussians:GaussianModel, mask):
    if mask is None:
        return
    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._opacity = gaussians._opacity[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]

def move_gaussians(gaussians:GaussianModel, mask, offset):
    if mask is None:
        return
    with torch.no_grad():
        gaussians._xyz[mask] += torch.tensor(offset, device=gaussians._xyz.device).reshape(1,3)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true", default=True)
    parser.add_argument("--skip_test", action="store_true", default=True)
    parser.add_argument("--skip_mesh", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true", default=False)
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    mask = None
    # mask = np.load("output/bicycle/mask/mask.npy")
    # mask_gaussians(gaussians, mask)
    move_gaussians(gaussians, mask, offset=[2, -1.2, 0.3])
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    

    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj' if mask is None else 'traj_mask', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj' if mask is None else 'render_traj_mask', 
                    num_frames=n_fames)
    else:
        print("render image ...")
        traj_dir = os.path.join(args.model_path, 'traj' if mask is None else 'traj_mask', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        # cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        # viewpoint_cam = cam_traj[0]
        viewpoint_cam = scene.getTrainCameras()[0]
        render_pkg = gaussExtractor.render(viewpoint_cam, gaussExtractor.gaussians)
        rgb = render_pkg['render']
        save_img_u8(rgb.detach().permute(1,2,0).cpu().numpy(), os.path.join(traj_dir, "color.png" if mask is None else "color_mask.png"))
        

# python render_video.py --model_path output/bicycle --source_path data/mipnerf360/indoor/bicycle -r 4 --render_path