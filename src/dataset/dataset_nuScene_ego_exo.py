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

# SEED4D spherical camera paths - these will be used as target views
SEED4D_DATASET_ROOT = '/app/data/seed4d/static/' # Change this to your data directory 
assert SEED4D_DATASET_ROOT is not None, "Update the location of the SEED4D Dataset"

@dataclass
class Dataset_NUSCENE_EGO_EXOCfg(DatasetCfgCommon):
    name: Literal["nuscene_ego_exo"]
    train_view_sampler: ViewSamplerCfg
    eval_view_sampler: ViewSamplerCfg
    max_fov: float
    z_near: float
    z_far: float
    # SEED4D compatibility
    training_towns: List[str] = None  # SEED4D towns to use for camera coordinates
    testing_towns: List[str] = None   
    selected_sensors: Optional[List[int]] = None  # List of SEED4D sensor indices to use
    
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
        
        # Configure sensor selection (for SEED4D context cameras)
        self.sensor_indices = self._configure_sensor_selection()
        
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
        
        # Get all samples (frames) from selected scenes
        self.samples = []
        for scene in self.scenes:
            sample_token = scene["first_sample_token"]
            while sample_token != "":
                sample = self.nusc.get("sample", sample_token)
                self.samples.append(sample_token)
                sample_token = sample["next"]
        
        # 🎯 SEED4D coordinate system setup (same as original SEED4D dataset)
        self._setup_seed4d_coordinate_system()
        
        # Configure resolutions
        self.context_resolution = (self.view_sampler.cfg.input_context_resolution, 
                                 self.view_sampler.cfg.input_context_resolution)
        self.target_resolution = (self.view_sampler.cfg.output_target_resolution, 
                                self.view_sampler.cfg.output_target_resolution)
        
        # Augmentation flag
        self.augment_flag = self.stage == 'train' and self.cfg.train_view_sampler.augment
        
        # Pre-load data into memory (with error handling)
        print(f"Pre-loading {len(self.samples)} NuScenes samples with SEED4D coordinates...")
        
        # Load all samples, not just first 3
        successful_loads = 0
        for idx in range(len(self.samples)):
            try:
                self.load_example_id(idx)
                successful_loads += 1
                if idx == 0:
                    print(f"✅ Successfully loaded first sample")
                elif idx % 20 == 0:  # Progress update every 20 samples
                    print(f"✅ Loaded {successful_loads}/{idx+1} samples...")
            except Exception as e:
                print(f"❌ Error loading sample {idx}: {e}")
                print(f"   Sample ID: {self.samples[idx] if idx < len(self.samples) else 'N/A'}")
                print(f"   SEED4D context transform: {self.selected_input_transform}")
                print(f"   SEED4D target transform: {self.selected_output_transform}")
                # Continue loading other samples instead of breaking
                continue
        
        print(f"✅ Successfully loaded {successful_loads}/{len(self.samples)} samples")
        
        print(f"NuScenes Dataset with SEED4D coordinates, initialized for {self.stage} stage")
        print(f"Will use {len(self.samples)} samples with augmentation = {self.augment_flag}")
        print(f"Selected SEED4D context sensors: {self.sensor_indices}")
        print(f"Using SEED4D towns: {self.cfg.training_towns or ['02']}")
    
    def _configure_sensor_selection(self) -> List[int]:
        """Configure which SEED4D sensors to use for context cameras."""
        if self.cfg.selected_sensors is not None:
            sensor_indices = self.cfg.selected_sensors
            print(f"Using explicitly selected SEED4D sensors: {sensor_indices}")
        else:
            # Default: use all 7 SEED4D sensors (0-6)
            sensor_indices = list(range(7))  # 0, 1, 2, 3, 4, 5, 6
            print(f"Using default SEED4D sensors (all 7): {sensor_indices}")
        
        return sensor_indices
    
    def _setup_seed4d_coordinate_system(self):
        """Setup SEED4D coordinate system paths (same as original SEED4D dataset)."""
        data_dir_naming = '/ClearNoon/vehicle.audi.tt/'
        
        # Use configuration values or default fallback
        training_towns = self.cfg.training_towns if self.cfg.training_towns is not None else ['02']
        testing_towns = self.cfg.testing_towns if self.cfg.testing_towns is not None else ['02']
        
        if self.stage == 'train':
            towns = training_towns
        else:
            towns = testing_towns
            
        # Get SEED4D paths (exactly like original SEED4D dataset)
        self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in towns]
        self.spawn_dirs = [self._str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle') 
                          for spawns_dir in self.parent_dirs]
        self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
        
        # 🎯 KEY: Use SEED4D coordinate transforms for BOTH context and target cameras
        if self.stage == 'train':
            # Context cameras: Use SEED4D nuscenes_invisible transforms for camera coordinates
            self.input_transforms = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' 
                                   for spawn_dir in self.spawn_dirs]
            # Target cameras: Use SEED4D sphere transforms  
            self.output_transforms = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_train.json' 
                                    for spawn_dir in self.spawn_dirs]
        else:
            # Validation/test
            self.input_transforms = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' 
                                   for spawn_dir in self.spawn_dirs]
            self.output_transforms = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_test.json' 
                                    for spawn_dir in self.spawn_dirs]
        
        # For simplicity, use the first available SEED4D spawn
        self.selected_input_transform = self.input_transforms[0] if self.input_transforms else None
        self.selected_output_transform = self.output_transforms[0] if self.output_transforms else None
        
        print(f"🎯 SEED4D Coordinate System Setup:")
        print(f"   Context transforms: {len(self.input_transforms)} available")
        print(f"   Target transforms: {len(self.output_transforms)} available")
        print(f"   Using context: {self.selected_input_transform}")
        print(f"   Using target: {self.selected_output_transform}")
    
    def _str_list_concat(self, pre_string, folders_dir, post_string):
        """Helper function to concatenate directory paths."""
        try:
            list_dirs = os.listdir(folders_dir)
            return [pre_string + directory + '/' + post_string for directory in list_dirs]
        except:
            return []
    
    def _modify_camera_index_3(self, extrinsics):
        """Add 0.8 to camera with index 3 Y-axis right after reading values."""
        if len(extrinsics) > 3:  # Check if camera index 3 exists
            print(f"🔧 MODIFYING CAMERA INDEX 3: Adding 0.8 to Y-axis translation")
            print(f"   Original position: {extrinsics[3][:3, 3].numpy() if isinstance(extrinsics, torch.Tensor) else extrinsics[3][:3, 3]}")
            
            # Add 0.8 to the Y-axis translation component (position) of camera index 3
            extrinsics[3][1, 3] += 0.8  # Only Y-axis
            
            print(f"   Modified position: {extrinsics[3][:3, 3].numpy() if isinstance(extrinsics, torch.Tensor) else extrinsics[3][:3, 3]}")
        else:
            print(f"⚠️  Camera index 3 not available (only {len(extrinsics)} cameras)")
        
        return extrinsics
    
    def debug_coordinate_systems(self, context_extrinsics, target_extrinsics, sample_name=""):
        """Debug function to analyze SEED4D coordinate systems."""
        print(f"\n🔍 SEED4D COORDINATE SYSTEM ANALYSIS {sample_name}")
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
        
        print(f"📍 SEED4D CONTEXT CAMERAS ({len(context_positions)} cameras):")
        print(f"  Sample positions (first 3):")
        for i, pos in enumerate(context_positions[:3]):
            print(f"    Camera {i}: [{pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}]")
        print(f"  Position statistics:")
        print(f"    Mean: [{context_positions.mean(axis=0)[0]:8.2f}, {context_positions.mean(axis=0)[1]:8.2f}, {context_positions.mean(axis=0)[2]:8.2f}]")
        print(f"    Distance from origin: {np.linalg.norm(context_positions, axis=1).mean():.2f} ± {np.linalg.norm(context_positions, axis=1).std():.2f}")
        
        print(f"\n🎯 SEED4D TARGET CAMERAS ({len(target_positions)} cameras):")
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
        
        print(f"\n⚖️  COORDINATE SYSTEM COMPARISON:")
        print(f"  Context center: [{context_center[0]:8.2f}, {context_center[1]:8.2f}, {context_center[2]:8.2f}]")
        print(f"  Target center:  [{target_center[0]:8.2f}, {target_center[1]:8.2f}, {target_center[2]:8.2f}]")
        print(f"  Distance between centers: {center_distance:.2f} meters")
        
        if center_distance < 50:
            print(f"  ✅ GOOD: Both coordinate systems are well aligned!")
        else:
            print(f"  ⚠️  WARNING: Large distance between coordinate systems!")
        
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
        """Load and cache data for a sample using SEED4D coordinates."""
        example_id = self.get_example_id(index)
        
        if not hasattr(self, "all_texture_context"):
            self.all_texture_context = {}
            self.all_texture_target = {}
            self.intrinsics_context = {}
            self.intrinsics_target = {}
            self.extrinsics_context = {}
            self.extrinsics_target = {}
        
        if example_id not in self.all_texture_context.keys():
            print(f"\n🚗 LOADING SAMPLE: {example_id}")
            
            # Initialize storage for this example
            self.all_texture_context[example_id] = []
            self.all_texture_target[example_id] = []
            self.intrinsics_context[example_id] = []
            self.intrinsics_target[example_id] = []
            self.extrinsics_context[example_id] = []
            self.extrinsics_target[example_id] = []
            
            # 🎯 LOAD SEED4D CAMERA COORDINATES (for context cameras)
            if self.selected_input_transform and os.path.exists(self.selected_input_transform):
                print(f"📐 Loading SEED4D context coordinates from: {self.selected_input_transform}")
                seed4d_context_paths, seed4d_context_intrinsics, seed4d_context_extrinsics = \
                    readPixelSplatCamera(self.selected_input_transform, 
                                       resolution=self.view_sampler.cfg.input_context_resolution,
                                       near=self.cfg.z_near, far=self.cfg.z_far)
                
                # 🔧 MODIFY CAMERA INDEX 3 RIGHT AFTER READING VALUES
                if isinstance(seed4d_context_extrinsics, list):
                    seed4d_context_extrinsics = torch.stack(seed4d_context_extrinsics)
                seed4d_context_extrinsics = self._modify_camera_index_3(seed4d_context_extrinsics)
                
                # Filter to selected sensors
                seed4d_context_paths = [seed4d_context_paths[i] for i in self.sensor_indices 
                                      if i < len(seed4d_context_paths)]
                if isinstance(seed4d_context_intrinsics, list):
                    seed4d_context_intrinsics = [seed4d_context_intrinsics[i] for i in self.sensor_indices 
                                               if i < len(seed4d_context_intrinsics)]
                    selected_indices = [i for i in self.sensor_indices if i < len(seed4d_context_extrinsics)]
                    seed4d_context_extrinsics = seed4d_context_extrinsics[selected_indices]
                else:
                    # Handle tensor case
                    selected_indices = [i for i in self.sensor_indices if i < len(seed4d_context_intrinsics)]
                    seed4d_context_intrinsics = seed4d_context_intrinsics[selected_indices]
                    seed4d_context_extrinsics = seed4d_context_extrinsics[selected_indices]
                
                print(f"   Selected {len(seed4d_context_paths)} SEED4D context cameras")
            else:
                print(f"❌ SEED4D context transform not found: {self.selected_input_transform}")
                return
            
            # 🚗 LOAD NUSCENES IMAGES (but use SEED4D coordinates)
            sample = self.nusc.get("sample", example_id)
            nuscenes_image_paths = []
            
            # Get NuScenes image paths for the selected sensors
            available_cameras = min(len(self.nuscenes_cameras), len(seed4d_context_paths))
            
            for i in range(available_cameras):
                if i < len(self.nuscenes_cameras):
                    camera_name = self.nuscenes_cameras[i]
                    if camera_name in sample["data"]:
                        camera_token = sample["data"][camera_name]
                        camera_data = self.nusc.get("sample_data", camera_token)
                        image_path = os.path.join(NUSCENE_DATA_DIR, camera_data["filename"])
                        nuscenes_image_paths.append(image_path)
                        print(f"   📷 NuScenes {camera_name}: {camera_data['filename']}")
                    else:
                        print(f"   ⚠️  Missing camera {camera_name} in sample")
                        nuscenes_image_paths.append(None)  # Placeholder
            
            # Ensure we have the right number of images to match SEED4D coordinates
            while len(nuscenes_image_paths) < len(seed4d_context_paths):
                # Duplicate last available image if needed
                if nuscenes_image_paths and nuscenes_image_paths[-1] is not None:
                    nuscenes_image_paths.append(nuscenes_image_paths[-1])
                else:
                    nuscenes_image_paths.append(None)
            
            # Store context data: NuScenes images + SEED4D coordinates
            self.all_texture_context[example_id] = nuscenes_image_paths[:len(seed4d_context_paths)]
            
            if isinstance(seed4d_context_intrinsics, list):
                self.intrinsics_context[example_id] = torch.stack(seed4d_context_intrinsics)
                self.extrinsics_context[example_id] = seed4d_context_extrinsics
            else:
                self.intrinsics_context[example_id] = seed4d_context_intrinsics
                self.extrinsics_context[example_id] = seed4d_context_extrinsics
            
            print(f"   ✅ Context: {len(self.all_texture_context[example_id])} NuScenes images with SEED4D coordinates")
            
            # 🎯 LOAD SEED4D TARGET CAMERAS (spherical views)
            if self.selected_output_transform and os.path.exists(self.selected_output_transform):
                print(f"🎯 Loading SEED4D target coordinates from: {self.selected_output_transform}")
                target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = \
                    readPixelSplatCamera(self.selected_output_transform, 
                                       resolution=self.view_sampler.cfg.output_target_resolution,
                                       near=self.cfg.z_near, far=self.cfg.z_far)
                
                # 🔧 MODIFY CAMERA INDEX 3 FOR TARGET CAMERAS TOO (if needed)
                if isinstance(target_extrinsics_matrices, list):
                    target_extrinsics_matrices = torch.stack(target_extrinsics_matrices)
                target_extrinsics_matrices = self._modify_camera_index_3(target_extrinsics_matrices)
                
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
                
                print(f"   ✅ Target: {len(target_image_paths)} SEED4D spherical cameras")
                
                # 🔍 DEBUG: Analyze coordinate systems
                self.debug_coordinate_systems(
                    self.extrinsics_context[example_id],
                    self.extrinsics_target[example_id], 
                    f"(Sample {index})"
                )
                
            else:
                print(f"❌ SEED4D target transform not found: {self.selected_output_transform}")
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
            print(f"⚠️ Sample {example_id} not in cache, loading on-demand...")
            self.load_example_id(index)
        
        print(f"\n📊 GETITEM DEBUG (index {index}):")
        print(f"  Available context views: {len(self.all_texture_context[example_id])}")
        print(f"  Available target views: {len(self.all_texture_target[example_id])}")
        
        # Sample views using the same strategy as SEED4D
        index_context, index_target = self.view_sampler.sample("SEED", 
                                                              self.extrinsics_context[example_id], 
                                                              self.extrinsics_target[example_id])

        print(f"  Sampled context indices: {index_context}")
        print(f"  Sampled target indices: {index_target}")
        
        # Load context images (NuScenes images with SEED4D coordinates)
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