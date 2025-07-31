import os
import json
import random
import itertools
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from numpy.random import default_rng
from io import BytesIO
from pathlib import Path
from typing import Literal, List, Optional, Union
import open3d as o3d

import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerCfg

from .dataset_readers import readPixelSplatCamera
from ..misc.general_utils import img_path_to_Torch, depth_path_to_Torch

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion.quaternion import Quaternion

NUSCENE_DATA_DIR = "/app/datasets/nuscenes_full/" 
assert NUSCENE_DATA_DIR is not None, "Update the location of the NUSCENE Dataset"

# Fixed SEED4D transform paths for target cameras only
SEED4D_TARGET_TRANSFORM = '/app/code/seed4d/data_analysis/data_seed4d/Town01/ClearNoon/vehicle.audi.tt/spawn_point_1/step_0/ego_vehicle/sphere_invisible/transforms/transforms_ego.json'

################################################################################################
##################### Camera Pose Transformation Functions #####################

def apply_coordinate_transformation(position):
    """
    Apply coordinate transformation:
    x_new = -z_old
    y_new = x_old
    z_new = -y_old
    """
    x_old, y_old, z_old = position
    return np.array([-z_old, x_old, -y_old])

def apply_rotation_transformation(quaternion):
    """
    Apply rotation transformation to match the coordinate system change
    """
    # Convert to rotation matrix
    rotation_matrix = quaternion.rotation_matrix
    
    # Transformation matrix
    T = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0, -1,  0]
    ])
    
    # Apply transformation: R_new = T * R_old
    transformed_rotation_matrix = T @ rotation_matrix
    
    # Convert back to quaternion
    transformed_quaternion = Quaternion(matrix=transformed_rotation_matrix)
    
    return transformed_quaternion

def quaternion_to_transform_matrix(quaternion, translation):
    """
    Convert quaternion and translation to 4x4 transformation matrix
    """
    rotation_matrix = quaternion.rotation_matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def reorder_camera_data(camera_data):
    """
    Reorder camera data from original order (0,1,2,3,4,5) to new order (0,1,5,3,4,2)
    """
    reorder_mapping = {0: 0, 1: 1, 2: 5, 3: 3, 4: 4, 5: 2}
    camera_list = list(camera_data.items())
    reordered_camera_data = {}
    
    for new_idx in range(len(camera_list)):
        if new_idx < len(camera_list):
            old_idx = reorder_mapping[new_idx]
            if old_idx < len(camera_list):
                camera_name, data = camera_list[old_idx]
                reordered_camera_data[camera_name] = data
    
    return reordered_camera_data

def transform_camera_poses(camera_input_data):
    """
    Transform camera pose data and return the transformed transforms JSON structure.
    
    Args:
        camera_input_data: Dictionary with camera data containing:
            - For each camera: {
                'translation': [x, y, z],
                'rotation': [w, x, y, z] or Quaternion object,
                'camera_intrinsic': 3x3 matrix or list,
                'image_width': int (optional, default 1600),
                'image_height': int (optional, default 900)
              }
    
    Returns:
        dict: Transformed transforms JSON structure
    """
    
    # Process each camera
    camera_data = {}
    for camera_name, data in camera_input_data.items():
        # Get original position and rotation
        original_translation = np.array(data['translation'])
        
        # Handle rotation input (could be quaternion object or list)
        if isinstance(data['rotation'], Quaternion):
            original_quaternion = data['rotation']
        else:
            # Assume [w, x, y, z] format
            rot = data['rotation']
            original_quaternion = Quaternion(w=rot[0], x=rot[1], y=rot[2], z=rot[3])
        
        # Apply x-axis flip
        x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
        original_quaternion = original_quaternion * x_axis_flip
        
        # Apply coordinate transformation
        transformed_translation = apply_coordinate_transformation(original_translation)
        transformed_quaternion = apply_rotation_transformation(original_quaternion)
        
        # Create transformation matrix
        transformed_transform_matrix = quaternion_to_transform_matrix(
            transformed_quaternion, transformed_translation
        )
        
        # Get camera intrinsics
        camera_intrinsic = np.array(data['camera_intrinsic'])
        fl_x = camera_intrinsic[0, 0]
        fl_y = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        
        # Get image dimensions (with defaults)
        w = data.get('image_width', 1600)
        h = data.get('image_height', 900)
        
        camera_data[camera_name] = {
            'transform_matrix': transformed_transform_matrix,
            'fl_x': float(fl_x),
            'fl_y': float(fl_y),
            'cx': float(cx),
            'cy': float(cy),
            'w': w,
            'h': h
        }
    
    # Reorder cameras
    camera_data = reorder_camera_data(camera_data)
    
    # Create transformed transforms JSON structure
    transformed_json = {
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }
    
    # Process each camera in the reordered data
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        frame = {
            "file_path": f"../sensors/{idx}_rgb.png",
            "depth_file_path": f"../sensors/{idx}_depth.png",
            "semantic_segmentation_file_path": f"../sensors/{idx}_semantic_segmentation.png",
            "instance_segmentation_file_path": f"../sensors/{idx}_instance_segmentation.png",
            "transform_matrix": data['transform_matrix'].tolist(),
            "fl_x": data['fl_x'],
            "fl_y": data['fl_y'],
            "cx": data['cx'],
            "cy": data['cy'],
            "w": data['w'],
            "h": data['h'],
            "camera_name": camera_name
        }
        
        transformed_json["frames"].append(frame)
    
    return transformed_json

################################################################################################

@dataclass
class Dataset_NUSCENE_EGO_EXOCfg(DatasetCfgCommon):
    name: Literal["nuscene_ego_exo"]
    train_view_sampler: ViewSamplerCfg
    eval_view_sampler: ViewSamplerCfg
    max_fov: float
    z_near: float
    z_far: float
    use_ego_pose: bool = False  # New flag to control ego pose usage
    
class Dataset_NUSCENE_EGO_EXO(Dataset):
    cfg: Dataset_NUSCENE_EGO_EXOCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: Dataset_NUSCENE_EGO_EXOCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        # Configure sensor selection (always use first 6 sensors for 6 NuScenes cameras)
        self.sensor_indices = list(range(6))  # Always use indices 0-5 for 6 NuScenes cameras
        
        # NuScenes setup
        self.version = 'v1.0-mini'  # or get from config
        self.nusc = NuScenes(version=self.version, dataroot=NUSCENE_DATA_DIR)
        
        # NuScenes camera names in order (for image loading)
        self.nuscenes_cameras = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        # Get scenes for current stage
        all_splits = create_splits_scenes()
        if self.version == 'v1.0-mini':
            self.usable_splits = {
                'train': all_splits['mini_train'],
                'val': all_splits['mini_val']
            }
        elif self.version == 'v1.0-trainval':
            self.usable_splits = {
                'train': all_splits['train'],
                'val': all_splits['val']
            }
        else:
            raise NotImplementedError(f"Version {self.version} not supported")
        
        # Select scenes based on stage
        if self.stage == 'train':
            scene_names = self.usable_splits["train"]
        elif self.stage == 'val':
            scene_names = self.usable_splits["val"] 
        elif self.stage == 'test':
            scene_names = self.usable_splits.get("test", self.usable_splits["val"])
        else:
            raise ValueError("Invalid stage")
        
        # Get all scenes for the stage
        self.scenes = []
        for scene in self.nusc.scene:
            if scene["name"] in scene_names:
                self.scenes.append(scene)
        
        print(f"Using {len(self.scenes)} scenes")
        
        # Get all samples (frames) from selected scenes
        self.samples = []
        for scene in self.scenes:
            sample_token = scene["first_sample_token"]
            while sample_token != "":
                sample = self.nusc.get("sample", sample_token)
                self.samples.append(sample_token)
                sample_token = sample["next"]
        
        # SEED4D coordinate system setup (using fixed target path only)
        self._setup_seed4d_coordinate_system()
        
        # Configure resolutions
        if self.stage == 'train':
            view_sampler_cfg = self.cfg.train_view_sampler
        else:
            view_sampler_cfg = self.cfg.eval_view_sampler
            
        self.context_resolution = (view_sampler_cfg.input_context_resolution, 
                                 view_sampler_cfg.input_context_resolution)
        self.target_resolution = (view_sampler_cfg.output_target_resolution, 
                                view_sampler_cfg.output_target_resolution)
        
        print(f"Using {self.stage} view sampler config:")
        print(f"   num_context_views: {view_sampler_cfg.num_context_views}")
        print(f"   num_target_views: {view_sampler_cfg.num_target_views}")
        print(f"   input_context_resolution: {view_sampler_cfg.input_context_resolution}")
        print(f"   output_target_resolution: {view_sampler_cfg.output_target_resolution}")
        
        # Augmentation flag
        self.augment_flag = self.stage == 'train' and self.cfg.train_view_sampler.augment
        
        # Pre-load data into memory (with error handling)
        print(f"Pre-loading {len(self.samples)} NuScenes samples with transformed coordinates...")
        
        # Load all samples
        successful_loads = 0
        for idx in range(len(self.samples)):
            try:
                self.load_example_id(idx)
                successful_loads += 1
                if idx == 0:
                    print(f"Successfully loaded first sample")
                elif idx % 20 == 0:  # Progress update every 20 samples
                    print(f"Loaded {successful_loads}/{idx+1} samples...")
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                print(f"   Sample ID: {self.samples[idx] if idx < len(self.samples) else 'N/A'}")
                # Continue loading other samples instead of breaking
                continue
        
        print(f"Successfully loaded {successful_loads}/{len(self.samples)} samples")
        
        print(f"NuScenes Dataset with transformed coordinates, initialized for {self.stage} stage")
        print(f"Will use {len(self.samples)} samples with augmentation = {self.augment_flag}")
        print(f"Using 6 NuScenes cameras with transformation: {self.sensor_indices}")
        print(f"Ego pose usage: {'ENABLED' if self.cfg.use_ego_pose else 'DISABLED (calibrated_sensor only)'}")
    
    def _setup_seed4d_coordinate_system(self):
        """Setup SEED4D coordinate system using fixed target path only."""
        
        # Only use fixed SEED4D coordinate transforms for target cameras
        self.selected_output_transform = SEED4D_TARGET_TRANSFORM
        
        print(f"SEED4D Coordinate System Setup:")
        print(f"   Context: Will read from NuScenes and transform")
        print(f"   Target transform: {self.selected_output_transform}")
        
        # Verify target file exists
        if not os.path.exists(self.selected_output_transform):
            print(f"WARNING: Target transform file not found: {self.selected_output_transform}")
        else:
            print(f"   Target transform file exists")
    
    def _extract_nuscenes_camera_data(self, sample_token):
        """
        Extract camera data from NuScenes sample in the same way as dataset_nuScene.py
        """
        sample = self.nusc.get("sample", sample_token)
        sample_frame_info = sample["data"]
        
        camera_input_data = {}
        
        print(f"Extracting camera data with ego_pose: {'ENABLED' if self.cfg.use_ego_pose else 'DISABLED'}")
        
        for camera_name in self.nuscenes_cameras:
            if camera_name not in sample_frame_info:
                print(f"Warning: Camera {camera_name} not found in sample")
                continue
                
            # Get sensor data
            sensor_data = self.nusc.get("sample_data", sample_frame_info[camera_name])
            
            # Get sensor pose information (calibrated_sensor)
            sensor_pose_information = self.nusc.get(table_name="calibrated_sensor", 
                                                  token=sensor_data["calibrated_sensor_token"])
            sensor_intrinsic_matrix = np.array(sensor_pose_information["camera_intrinsic"])
            sensor_pose_rotation = Quaternion(sensor_pose_information["rotation"])
            sensor_pose_translation = np.array(sensor_pose_information["translation"])
            
            # Create sensor transform matrix (sensor relative to ego vehicle)
            sensor_transform_matrix = transform_matrix(sensor_pose_translation, sensor_pose_rotation)
            
            # Conditionally include ego pose
            if self.cfg.use_ego_pose:
                # Get ego pose information (vehicle position in world)
                ego_pose_information = self.nusc.get(table_name="ego_pose", token=sensor_data["ego_pose_token"])
                ego_pose_rotation = Quaternion(ego_pose_information["rotation"])
                ego_pose_translation = np.array(ego_pose_information["translation"])
                ego_transform_matrix = transform_matrix(ego_pose_translation, ego_pose_rotation)
                
                # Combine ego pose and sensor calibration
                final_transform_matrix = ego_transform_matrix @ sensor_transform_matrix
                print(f"   {camera_name}: Using ego_pose + calibrated_sensor")
            else:
                # Use only sensor calibration (relative to ego vehicle)
                final_transform_matrix = sensor_transform_matrix
                print(f"   {camera_name}: Using calibrated_sensor only (relative to ego)")
            
            # Extract translation and rotation for transformation
            translation = final_transform_matrix[:3, 3]
            rotation_matrix = final_transform_matrix[:3, :3]
            rotation_quaternion = Quaternion(matrix=rotation_matrix)
            
            # Store in format expected by transform_camera_poses
            camera_input_data[camera_name] = {
                'translation': translation.tolist(),
                'rotation': [rotation_quaternion.w, rotation_quaternion.x, rotation_quaternion.y, rotation_quaternion.z],
                'camera_intrinsic': sensor_intrinsic_matrix.tolist(),
                'image_width': sensor_data.get("width", 1600),
                'image_height': sensor_data.get("height", 900),
                'image_path': os.path.join(NUSCENE_DATA_DIR, sensor_data["filename"])
            }
        
        return camera_input_data

    
    def save_transformed_json_debug(self, transformed_json, sample_token, output_dir='debug_transforms'):
        """
        Save transformed JSON data as debug files, similar to the original camera_pose_transformation_marius.py
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Include ego pose flag in filename for clarity
        ego_flag = "with_ego" if self.cfg.use_ego_pose else "no_ego"
        transformed_json_path = os.path.join(output_dir, f'transforms_transformed_{ego_flag}_{sample_token[:8]}.json')
        
        print(f"Saving transformed JSON debug file...")
        
        # Save transformed JSON file
        with open(transformed_json_path, 'w') as f:
            json.dump(transformed_json, f, indent=4)
        
        print(f"Transformed transforms JSON saved to: {transformed_json_path}")
        
        # Print summary of transformation
        print(f"\nTRANSFORMED JSON SUMMARY for sample {sample_token[:8]} ({'with ego pose' if self.cfg.use_ego_pose else 'calibrated sensor only'}):")
        print(f"  Camera model: {transformed_json['camera_model']}")
        print(f"  Number of frames: {len(transformed_json['frames'])}")
        print(f"  Coordinate system: {'Global (ego + sensor)' if self.cfg.use_ego_pose else 'Ego-relative (sensor only)'}")
        for i, frame in enumerate(transformed_json['frames']):
            transform_matrix = np.array(frame['transform_matrix'])
            position = transform_matrix[:3, 3]
            print(f"  Frame {i} ({frame['camera_name']}): pos=[{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:6.2f}]")
        
        return transformed_json_path

    def debug_coordinate_systems(self, context_extrinsics, target_extrinsics, sample_name=""):
        """Debug function to analyze coordinate systems."""
        print(f"\nCOORDINATE SYSTEM ANALYSIS {sample_name}")
        print("=" * 60)
        
        # Extract positions (translation components)
        if isinstance(context_extrinsics, torch.Tensor):
            context_positions = context_extrinsics[:, :3, 3].numpy()
        else:
            context_positions = context_extrinsics[:, :3, 3]
            
        if isinstance(target_extrinsics, torch.Tensor):
            target_positions = target_extrinsics[:, :3, 3].numpy()
        else:
            target_positions = target_extrinsics[:, :3, 3]
        
        print(f"TRANSFORMED CONTEXT CAMERAS ({len(context_positions)} cameras):")
        print(f"  Sample positions (first 3):")
        for i, pos in enumerate(context_positions[:3]):
            print(f"    Camera {i}: [{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}]")
        print(f"  Position statistics:")
        print(f"    Mean: [{context_positions.mean(axis=0)[0]:8.2f}, {context_positions.mean(axis=0)[1]:8.2f}, {context_positions.mean(axis=0)[2]:8.2f}]")
        print(f"    Distance from origin: {np.linalg.norm(context_positions, axis=1).mean():.2f} ± {np.linalg.norm(context_positions, axis=1).std():.2f}")
        
        print(f"\nSEED4D TARGET CAMERAS ({len(target_positions)} cameras):")
        print(f"  Sample positions (first 3):")
        for i, pos in enumerate(target_positions[:3]):
            print(f"    Camera {i}: [{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}]")
        print(f"  Position statistics:")
        print(f"    Mean: [{target_positions.mean(axis=0)[0]:8.2f}, {target_positions.mean(axis=0)[1]:8.2f}, {target_positions.mean(axis=0)[2]:8.2f}]")
        print(f"    Distance from origin: {np.linalg.norm(target_positions, axis=1).mean():.2f} ± {np.linalg.norm(target_positions, axis=1).std():.2f}")
        
        # Distance between coordinate systems
        context_center = context_positions.mean(axis=0)
        target_center = target_positions.mean(axis=0)
        center_distance = np.linalg.norm(context_center - target_center)
        
        print(f"\nCOORDINATE SYSTEM COMPARISON:")
        print(f"  Context center: [{context_center[0]:8.2f}, {context_center[1]:8.2f}, {context_center[2]:8.2f}]")
        print(f"  Target center:  [{target_center[0]:8.2f}, {target_center[1]:8.2f}, {target_center[2]:8.2f}]")
        print(f"  Distance between centers: {center_distance:.2f} meters")
        
        if center_distance < 50:
            print(f"  GOOD: Both coordinate systems are well aligned!")
        else:
            print(f"  WARNING: Large distance between coordinate systems!")
        
        print("=" * 60)
    
    def __len__(self):
        return len(self.samples)
    
    def get_bound(self, bound: Literal["z_near", "z_far", "fov"], num_views: int) -> Float[Tensor, " view"]:
        """Return near and far bounds with shape = (num_views,)"""
        if bound == 'z_near': 
            value = torch.tensor(self.cfg.z_near, dtype=torch.float32) 
        elif bound == 'z_far': 
            value = torch.tensor(self.cfg.z_far, dtype=torch.float32) 
        elif bound == 'fov': 
            value = torch.tensor(self.cfg.max_fov, dtype=torch.float32) 
        else: 
            raise KeyError("Wrong bound type is passed to retrieve")
        return repeat(value, "-> v", v=num_views)

    def get_example_id(self, index):
        """Get example ID for the sample."""
        return self.samples[index]
    
    def load_example_id(self, index):
        """Load and cache data for a sample using transformed NuScenes coordinates."""
        example_id = self.get_example_id(index)
        
        if not hasattr(self, "all_texture_context"):
            self.all_texture_context = {}
            self.all_texture_target = {}
            self.intrinsics_context = {}
            self.intrinsics_target = {}
            self.extrinsics_context = {}
            self.extrinsics_target = {}
        
        if example_id not in self.all_texture_context.keys():
            print(f"\nLOADING SAMPLE: {example_id}")
            
            # Initialize storage for this example
            self.all_texture_context[example_id] = []
            self.all_texture_target[example_id] = []
            self.intrinsics_context[example_id] = []
            self.intrinsics_target[example_id] = []
            self.extrinsics_context[example_id] = []
            self.extrinsics_target[example_id] = []
            
            # EXTRACT NUSCENES CAMERA DATA (same as dataset_nuScene.py)
            print(f"Extracting NuScenes camera data...")
            camera_input_data = self._extract_nuscenes_camera_data(example_id)
            
            # TRANSFORM CAMERA POSES using the transformation function
            print(f"Applying coordinate transformation...")
            transformed_json = transform_camera_poses(camera_input_data)
            
            # DEBUG: Save transformed JSON file
            self.save_transformed_json_debug(transformed_json, example_id)
### DEBUG: until here everything should be okay ###


####copied code from hardcoded version start####

            self.selected_input_transform = self.save_transformed_json_debug(transformed_json, example_id) #match between above code and pasted code from hardcoded version

            if self.selected_input_transform and os.path.exists(self.selected_input_transform):
                print(f"Loading SEED4D context coordinates from: {self.selected_input_transform}")
                seed4d_context_paths, seed4d_context_intrinsics, seed4d_context_extrinsics = \
                    readPixelSplatCamera(self.selected_input_transform, 
                                       resolution=self.context_resolution[0],
                                       near=self.cfg.z_near, far=self.cfg.z_far)
                
                # Convert to tensor if needed
                if isinstance(seed4d_context_extrinsics, list):
                    seed4d_context_extrinsics = torch.stack(seed4d_context_extrinsics)
                
                # Filter to first 6 sensors to match 6 NuScenes cameras (indices 0-5)
                valid_sensor_indices = list(range(6))  # Always use 0, 1, 2, 3, 4, 5
                
                #print(f"   Using first 6 SEED4D sensors: {valid_sensor_indices}")
                
                seed4d_context_paths = [seed4d_context_paths[i] for i in valid_sensor_indices if i < len(seed4d_context_paths)]
                
                if isinstance(seed4d_context_intrinsics, list):
                    seed4d_context_intrinsics = [seed4d_context_intrinsics[i] for i in valid_sensor_indices if i < len(seed4d_context_intrinsics)]
                    seed4d_context_extrinsics = seed4d_context_extrinsics[[i for i in valid_sensor_indices if i < len(seed4d_context_extrinsics)]]
                else:
                    # Handle tensor case
                    valid_indices = [i for i in valid_sensor_indices if i < len(seed4d_context_intrinsics)]
                    seed4d_context_intrinsics = seed4d_context_intrinsics[valid_indices]
                    seed4d_context_extrinsics = seed4d_context_extrinsics[valid_indices]
                
                print(f"   Selected {len(seed4d_context_paths)} SEED4D context cameras")
            else:
                print(f"SEED4D context transform not found: {self.selected_input_transform}")
                return
            
            # LOAD NUSCENES IMAGES (but use SEED4D coordinates)
            sample = self.nusc.get("sample", example_id)
            nuscenes_image_paths = []
            
            # Get NuScenes image paths - exactly 6 cameras to match 6 SEED4D sensors
            for i in range(6):  # Always use exactly 6 cameras
                if i < len(self.nuscenes_cameras):
                    camera_name = self.nuscenes_cameras[i]
                    if camera_name in sample["data"]:
                        camera_token = sample["data"][camera_name]
                        camera_data = self.nusc.get("sample_data", camera_token)
                        image_path = os.path.join(NUSCENE_DATA_DIR, camera_data["filename"])
                        nuscenes_image_paths.append(image_path)
                        print(f"   NuScenes {camera_name}: {camera_data['filename']}")
                    else:
                        print(f"   Missing camera {camera_name} in sample")
                        # Use a black image placeholder if camera is missing
                        nuscenes_image_paths.append(None)
                else:
                    # This shouldn't happen since we have exactly 6 NuScenes cameras
                    print(f"   Error: Trying to access camera index {i} but only have {len(self.nuscenes_cameras)} cameras")
                    nuscenes_image_paths.append(None)
            
            # Store context data: NuScenes images + SEED4D coordinates
            self.all_texture_context[example_id] = nuscenes_image_paths
            print('nuscenes_image_paths', nuscenes_image_paths)
            print('seed4d_context_paths', seed4d_context_paths)
            #self.all_texture_context[example_id] = seed4d_context_paths


            
            if isinstance(seed4d_context_intrinsics, list):
                self.intrinsics_context[example_id] = torch.stack(seed4d_context_intrinsics)
                self.extrinsics_context[example_id] = seed4d_context_extrinsics
            else:
                self.intrinsics_context[example_id] = seed4d_context_intrinsics
                self.extrinsics_context[example_id] = seed4d_context_extrinsics

            
            # Ensure extrinsics are proper tensors with correct shape
            if not isinstance(self.extrinsics_context[example_id], torch.Tensor):
                self.extrinsics_context[example_id] = torch.tensor(self.extrinsics_context[example_id], dtype=torch.float32)
            
            print(f"   Context extrinsics shape: {self.extrinsics_context[example_id].shape}")
            print(f"   Context: {len(self.all_texture_context[example_id])} NuScenes images with SEED4D coordinates")



####copied code from hardcoded version end####         
            




            '''
            # Extract transformed data
            transformed_frames = transformed_json["frames"]
            
            # Process transformed context data
            nuscenes_image_paths = []
            transformed_intrinsics = []
            transformed_extrinsics = []
            
            for frame_data in transformed_frames:
                # Get image path from original camera data
                camera_name = frame_data["camera_name"]
                if camera_name in camera_input_data:
                    image_path = camera_input_data[camera_name]["image_path"]
                    nuscenes_image_paths.append(image_path)
                    print(f"   {camera_name}: {os.path.basename(image_path)}")
                else:
                    nuscenes_image_paths.append(None)
                
                # Extract intrinsics
                intrinsic_matrix = torch.tensor([
                    [frame_data["fl_x"], 0, frame_data["cx"]],
                    [0, frame_data["fl_y"], frame_data["cy"]],
                    [0, 0, 1]
                ], dtype=torch.float32)
                transformed_intrinsics.append(intrinsic_matrix)
                
                # Extract extrinsics
                extrinsic_matrix = torch.tensor(frame_data["transform_matrix"], dtype=torch.float32)
                transformed_extrinsics.append(extrinsic_matrix)
            
            # Convert to tensors
            transformed_intrinsics = torch.stack(transformed_intrinsics)
            transformed_extrinsics = torch.stack(transformed_extrinsics)
            
            
            # Store context data: NuScenes images + Transformed coordinates
            self.all_texture_context[example_id] = nuscenes_image_paths
            self.intrinsics_context[example_id] = transformed_intrinsics
            self.extrinsics_context[example_id] = transformed_extrinsics
            
            print(f"   Context extrinsics shape: {self.extrinsics_context[example_id].shape}")
            print(f"   Context: {len(self.all_texture_context[example_id])} NuScenes images with transformed coordinates")
            '''
### DEBUG: from here everything should be okay ###     
            # LOAD SEED4D TARGET CAMERAS (spherical views)
            if self.selected_output_transform and os.path.exists(self.selected_output_transform):
                print(f"Loading SEED4D target coordinates from: {self.selected_output_transform}")
                target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = \
                    readPixelSplatCamera(self.selected_output_transform, 
                                       resolution=self.target_resolution[0],
                                       near=self.cfg.z_near, far=self.cfg.z_far)
                
                # Convert to tensor if needed
                if isinstance(target_extrinsics_matrices, list):
                    target_extrinsics_matrices = torch.stack(target_extrinsics_matrices)
                
                # Store target data
                self.all_texture_target[example_id] = target_image_paths
                
                if isinstance(target_intrinsics_matrices, list):
                    self.intrinsics_target[example_id] = torch.stack(target_intrinsics_matrices)
                    self.extrinsics_target[example_id] = target_extrinsics_matrices
                elif len(target_intrinsics_matrices.shape) == 3:
                    self.intrinsics_target[example_id] = target_intrinsics_matrices
                    self.extrinsics_target[example_id] = target_extrinsics_matrices
                elif len(target_intrinsics_matrices.shape) == 2:
                    self.intrinsics_target[example_id] = target_intrinsics_matrices.unsqueeze(0)
                    self.extrinsics_target[example_id] = target_extrinsics_matrices.unsqueeze(0)
                else:
                    self.intrinsics_target[example_id] = target_intrinsics_matrices
                    self.extrinsics_target[example_id] = target_extrinsics_matrices
                
                # Ensure extrinsics are proper tensors with correct shape
                if not isinstance(self.extrinsics_target[example_id], torch.Tensor):
                    self.extrinsics_target[example_id] = torch.tensor(self.extrinsics_target[example_id], dtype=torch.float32)
                
                print(f"   Target extrinsics shape: {self.extrinsics_target[example_id].shape}")
                print(f"   Target: {len(target_image_paths)} SEED4D spherical cameras")
                
                # DEBUG: Analyze coordinate systems
                self.debug_coordinate_systems(
                    self.extrinsics_context[example_id],
                    self.extrinsics_target[example_id], 
                    f"(Sample {index})"
                )
                
            else:
                print(f"SEED4D target transform not found: {self.selected_output_transform}")
                # Create dummy targets as fallback
                num_dummy_targets = 20
                self.all_texture_target[example_id] = [f"dummy_target_{i}.png" for i in range(num_dummy_targets)]
                dummy_intrinsic = torch.eye(3).float()
                dummy_extrinsic = torch.eye(4).float()
                self.intrinsics_target[example_id] = dummy_intrinsic.unsqueeze(0).repeat(num_dummy_targets, 1, 1)
                self.extrinsics_target[example_id] = dummy_extrinsic.unsqueeze(0).repeat(num_dummy_targets, 1, 1)
    
    def __getitem__(self, index):
        example_id = self.get_example_id(index)
        
        # Ensure the sample is loaded
        if example_id not in self.all_texture_context:
            print(f"Sample {example_id} not in cache, loading on-demand...")
            self.load_example_id(index)
        
        print(f"\nGETITEM DEBUG (index {index}):")
        print(f"  Available context views: {len(self.all_texture_context[example_id])}")
        print(f"  Available target views: {len(self.all_texture_target[example_id])}")
        print(f"  Stage: {self.stage}")
        print(f"  View sampler config - num_context_views: {self.view_sampler.cfg.num_context_views}")
        print(f"  View sampler config - num_target_views: {self.view_sampler.cfg.num_target_views}")
        
        # Debug extrinsics shapes before sampling
        context_extrinsics = self.extrinsics_context[example_id]
        target_extrinsics = self.extrinsics_target[example_id]
        #print(f"  Context extrinsics shape: {context_extrinsics.shape}")
        #print(f"  Target extrinsics shape: {target_extrinsics.shape}")
        
        # Check for NaN or infinite values that could cause probability issues
        if torch.isnan(context_extrinsics).any():
            print(f"  WARNING: Context extrinsics contains NaN values!")
        if torch.isinf(context_extrinsics).any():
            print(f"  WARNING: Context extrinsics contains infinite values!")
        if torch.isnan(target_extrinsics).any():
            print(f"  WARNING: Target extrinsics contains NaN values!")
        if torch.isinf(target_extrinsics).any():
            print(f"  WARNING: Target extrinsics contains infinite values!")
        
        # Sample views using the same strategy as SEED4D
        try:
            index_context, index_target = self.view_sampler.sample("SEED", 
                                                                  context_extrinsics, 
                                                                  target_extrinsics)
        except Exception as e:
            print(f"Error in view sampler: {e}")
            #print(f"Context extrinsics shape: {context_extrinsics.shape}")
            #print(f"Target extrinsics shape: {target_extrinsics.shape}")
            
            # Fallback: use all 6 context cameras and a small number of target cameras
            num_context = len(self.all_texture_context[example_id])  # Use all available context cameras
            num_target = min(2, len(self.all_texture_target[example_id]))  # Use 2 target cameras
            index_context = torch.arange(num_context)  # Use all 6 context cameras
            index_target = torch.arange(num_target)
            print(f"Using fallback sampling: context={index_context} (all {num_context} cameras), target={index_target}")

        print(f"  Sampled context indices: {index_context}")
        print(f"  Sampled target indices: {index_target}")
        
        # Load context images (NuScenes images with transformed coordinates)
        context_images = []
        for image_path in np.array(self.all_texture_context[example_id])[index_context.numpy()]:
            if image_path and os.path.exists(image_path):
                context_images.append(img_path_to_Torch(image_path, self.context_resolution))
            else:
                # Create black image as fallback
                context_images.append(torch.zeros(3, *self.context_resolution))
        
        context_images = torch.stack(context_images).float()
        context_extrinsics = self.extrinsics_context[example_id][index_context.numpy()]
        context_intrinsics = self.intrinsics_context[example_id][index_context.numpy()]
        
        # Load target images (SEED4D spherical cameras)
        if len(self.all_texture_target[example_id]) > 0 and not self.all_texture_target[example_id][0].startswith("dummy"):
            target_images = [img_path_to_Torch(image_path, self.target_resolution)
                           for image_path in np.array(self.all_texture_target[example_id])[index_target.numpy()]]
            target_images = torch.stack(target_images).float()
            
            # Load target depths
            target_depths = [depth_path_to_Torch(image_path[:image_path.rfind('_')] + "_depth.png", self.target_resolution)
                           for image_path in np.array(self.all_texture_target[example_id])[index_target.numpy()]]
            target_depths = torch.stack(target_depths).float()
        else:
            # Dummy targets if no SEED4D data available
            target_images = torch.zeros(len(index_target), 3, *self.target_resolution)
            target_depths = torch.zeros(len(index_target), *self.target_resolution)
        
        target_extrinsics = self.extrinsics_target[example_id][index_target.numpy()]
        target_intrinsics = self.intrinsics_target[example_id][index_target.numpy()]
        
        # Return data structure identical to SEED4D
        example = {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "image": context_images,
                "near": self.get_bound("z_near", len(index_context)),
                "far": self.get_bound("z_far", len(index_context)),
                "fov": self.get_bound("fov", len(index_context)),
                "index": index_context,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "depth": rearrange(target_depths, "v h w -> v 1 h w"),
                "near": self.get_bound("z_near", len(index_target)),
                "far": self.get_bound("z_far", len(index_target)),
                "fov": self.get_bound("fov", len(index_target)),
                "index": index_target,
            },
            "scene": "nuScene"
        }
        return example

################################################################################################
##################### Extra functions for processing files and directories #####################

def str_list_concat(pre_string, folders_dir, post_string):
    list_dirs = os.listdir(folders_dir)
    return [pre_string + directory + '/' + post_string for directory in list_dirs]

# copied from https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string
def find_nth_reverse(haystack: str, needle: str, n: int) -> int:
    end = haystack.rfind(needle)
    while end >= 0 and n > 1:
        end = haystack.rfind(needle, 0, end - len(needle))
        n -= 1
    return end