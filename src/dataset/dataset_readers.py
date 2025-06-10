### taken & modified from https://github.com/szymanowiczs/splatter-image/tree/a2cbd6df67b2e34d10c8808dc1e8fe0daf4c02c7

import os
import json
import torch
from PIL import Image
from typing import NamedTuple
from ..misc.graphics_utils import focal2fov, fov2focal, load_metadata
import numpy as np
from pathlib import Path

def readPixelSplatCamera(transform_path, resolution = None, near = 0.0, far = 70.0):
    # first read the transforms file for cameras
    with open(transform_path, 'r') as f: jsonData = json.load(f)
    # we are going to assume that images will have square shape with
    # side length equal to the width of the original image and constant focal length
    intrinsic_normal_fl_x = jsonData['fl_x'] * (resolution / jsonData['w'])
    intrinsic_normal_fl_y = jsonData['fl_y'] * (resolution / jsonData['h'])
    
    image_paths = []
    pose_bounds = []
    base_image_dir = transform_path[:transform_path.rfind('/', 0, transform_path.rfind('/'))]
    
    for frame_dicts in jsonData['frames']:
        frame_image_path = base_image_dir + '/' + frame_dicts['file_path'][2:]
        image_paths.append(frame_image_path)
        # Carla cameras are camera-to-world transforms
        # need to change from Carla camera axes (x right, y up, z back)
        # to the COLMAP format (x right, y down, z forward)
        c2w = np.array(frame_dicts['transform_matrix'])
        
        # extrinsics ==> [y-down, x-right, z-backwards]
        extrinsics = np.zeros((3, 5))     # as required in https://github.com/dcharatan/pixelsplat/issues/68
        extrinsics[:, 0] = - c2w[:3, 1]   # -y
        extrinsics[:, 1] = c2w[:3, 0]     #  x
        extrinsics[:, 2:4] = c2w[:3, 2:4] #  z and t 
        extrinsics[:, 4] = np.array([resolution, intrinsic_normal_fl_x, intrinsic_normal_fl_y])    # [resolution, fl_x, fl_y]
        # flatten each extrinsics matrix and concatenate with near and far depth values ==> get Nx17 matrix 
        pose_bounds.append(np.concatenate((extrinsics.flatten(), np.array([near, far]))))
        
    pose_bounds = torch.from_numpy(np.stack(pose_bounds))
    extrinsics_matrices, intrinsics_matrices = load_metadata(pose_bounds)
    return image_paths, intrinsics_matrices, extrinsics_matrices

    
    