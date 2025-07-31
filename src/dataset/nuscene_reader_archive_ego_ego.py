import os
import torch
from PIL import Image
from typing import NamedTuple
from ..misc.graphics_utils import load_metadata
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import numpy as np

# # # all cameras per frame will be read in the order below: 
desired_sensor_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'] 

# a map to determine which radars to be projected onto each camera image plane
CAM2RADARS = {
        'CAM_FRONT': ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT'],
        'CAM_FRONT_LEFT': ['RADAR_FRONT', 'RADAR_FRONT_LEFT',
                           'RADAR_FRONT_RIGHT'],
        'CAM_FRONT_RIGHT': ['RADAR_FRONT', 'RADAR_FRONT_LEFT',
                            'RADAR_FRONT_RIGHT'],
        'CAM_BACK_LEFT': ['RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT'],
        'CAM_BACK_RIGHT': ['RADAR_FRONT_RIGHT', 'RADAR_BACK_RIGHT'],
        'CAM_BACK': ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
        }

# for filtering bboxes of stationary objects
STATIONARY_CATEGORIES={'movable_object.trafficcone', 'movable_object.barrier',
                       'movable_object.debris', 'static_object.bicycle_rack'}

class CameraInfo(NamedTuple):
    ego_T: np.array
    intrinsic: np.array
    extrinsic: np.array
    image_path: str
    width: int
    height: int
    name: str
    
def frame_cameras_transform(frame_dictionary, nuscene_loader=None, resolution=16, near=0.0, far=10.0, return_pc=False):
    features = []
    extrinsics = []
    intrinsics = []
    pose_bounds = []
    frame_camera_parameter = frame_dictionary["frame_info"]
    for frame_info in frame_camera_parameter:
        image =  np.transpose(np.array(Image.open(frame_info.image_path).resize((resolution, resolution)), 
                                       dtype=float)/255.0, (2, 0, 1))    
        # # intrinsics
        intrinsic_normal = np.zeros((3,3)) 
        intrinsic_normal[0,0] = frame_info.intrinsic[0, 0] * (resolution / frame_info.width)
        intrinsic_normal[1,1] = frame_info.intrinsic[1, 1] * (resolution / frame_info.height)
        intrinsic_normal[2,2] = 1
        intrinsic_normal[0,2] = resolution / 2
        intrinsic_normal[1,2] = resolution / 2
        # #
        # Mapping from nuScene format (x forward, y left, z up)
        # to the COLMAP format (x right, y down, z forward)
        # c2w ==> [x-right, y-up, z-back] 
        rotation_matrix = frame_info.extrinsic[:3, :3]
        translation_vector = frame_info.extrinsic[:, 3]
        c2w = np.zeros((4, 4))    
        c2w[:3, 0] =   rotation_matrix[:, 0]  
        c2w[:3, 1] = - rotation_matrix[:, 1]   
        c2w[:3, 2] = - rotation_matrix[:, 2]  
        # c2w[:3, :3] = rotation_matrix
        c2w[:, 3] =   translation_vector
        # extrinsics ==> [y-down, x-right, z-backwards]
        extrinsics = np.zeros((3, 5))     
        extrinsics[:, 0] = - c2w[:3, 1]   # -y
        extrinsics[:, 1] = c2w[:3, 0]     #  x
        extrinsics[:, 2:4] = c2w[:3, 2:4] #  z and t 
        extrinsics[:, 4] = np.array([resolution, intrinsic_normal[0,0], intrinsic_normal[1,1]])    # [resolution, fl_x, fl_y]
        # flatten each extrinsics matrix and concatenate with near and far depth values ==> get Nx17 matrix 
        pose_bounds.append(np.concatenate((extrinsics.flatten(), np.array([near, far]))))
        features.append(torch.from_numpy(image))
    # # 
    pose_bounds = torch.from_numpy(np.stack(pose_bounds))
    extrinsics, intrinsics = load_metadata(pose_bounds)
    features = torch.unsqueeze(torch.stack(features), 0).type(torch.FloatTensor)
    # # # # # # # # # # # # # # # # # # # # # # 
    if return_pc == True:
        lidar_token = frame_dictionary["lidar_token"]
        pointsensor = nuscene_loader.get('sample_data', lidar_token)
        pcl_path = os.path.join(nuscene_loader.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nuscene_loader.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nuscene_loader.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))
        point_cloud = torch.from_numpy(pc.points.T)[:, :3].type(torch.FloatTensor)
        return point_cloud, features, torch.unsqueeze(extrinsics, dim=0), torch.unsqueeze(intrinsics, dim=0) 
    return None, features, torch.unsqueeze(extrinsics, dim=0), torch.unsqueeze(intrinsics, dim=0)
# # # 
def frame_seq_transform(frame_seq, nuscene_loader=None, resolution=16, 
                        near=0.007, far=60.0, nuscene_lidar_point_num=0):
    frame_seq_point_cloud = None
    frame_seq_features = []
    frame_seq_extrinsics = []
    frame_seq_intrinsics = []
    # # # middle element will always act as a reference and we will
    # # # retrieve LIDAR info for that frame
    # return_pc_flags = [idx==int(len(frame_seq)//2) for idx in range(len(frame_seq))]
    return_pc_flags = [idx==0 for idx in range(len(frame_seq))]
    for pc_flag, frame_camera_parameter in zip(return_pc_flags, frame_seq):
        point_cloud, features, extrinsics, intrinsics \
            = frame_cameras_transform(frame_camera_parameter, nuscene_loader, 
                                      resolution, near, far, pc_flag)
        if point_cloud != None: frame_seq_point_cloud = point_cloud
        frame_seq_features.append(features)
        frame_seq_extrinsics.append(extrinsics)
        frame_seq_intrinsics.append(intrinsics)
    # # # randomly sample fixed amount of points from
    # # # provided LIDAR data that will help during batching
    random_point_indexes = torch.randint(0, len(frame_seq_point_cloud), (nuscene_lidar_point_num,))
    points = frame_seq_point_cloud[random_point_indexes]
    features = torch.concatenate(frame_seq_features)
    extrinsics = torch.concatenate(frame_seq_extrinsics)
    intrinsics = torch.concatenate(frame_seq_intrinsics)
    return points, features, extrinsics, intrinsics