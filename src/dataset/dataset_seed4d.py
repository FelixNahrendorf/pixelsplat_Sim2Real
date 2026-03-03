'''
        This Dataloader is used for ego-ego, ego-exo training and evaluation on SEED4D dataset and ego-exo-mixed domain training on SEED4D and Nuscene Dataset
'''

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

# ============ ADDED IMPORTS ============
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion.quaternion import Quaternion
from .nuscene_reader import desired_sensor_names, CameraInfo
# =======================================

SEED4D_DATASET_ROOT = '/app/felix/data/seed4d/data/data_diverse_1600x900_2poses/static/'
assert SEED4D_DATASET_ROOT is not None, "Update the location of the SEED4D Dataset"

LIDAR_DATASET_ROOT = '/app/new/seed4d/pseudo_lidar/'

# ============ ADDED CONSTANT ============
NUSCENE_DATA_DIR = "/app/datasets/nuscenes_full/"
assert NUSCENE_DATA_DIR is not None, "Update the location of the NUSCENE Dataset"
# ========================================

@dataclass
class Dataset_SEED4DCfg(DatasetCfgCommon):
    name: Literal["seed4d"]
    train_view_sampler: ViewSamplerCfg
    eval_view_sampler: ViewSamplerCfg
    max_fov: float
    z_near: float
    z_far: float
    experiment: str
    training_towns: List[str] = None
    testing_towns: List[str] = None
    selected_sensors: Optional[List[int]] = None
    nuscene_scene_index: Optional[List[int]] = None


class Dataset_SEED4D(Dataset):
    cfg: Dataset_SEED4DCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: Dataset_SEED4DCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Lookup dicts mapping SEED4D example_id -> nuScenes sample token / sample_data tokens,
        # and image file path -> sample_data token (used for depth map retrieval)
        self.nuscene_token_per_example = {}
        self.nuscene_sample_data_tokens_per_example = {}
        self.nuscene_path_to_token = {}
        
        self.sensor_indices = self._configure_sensor_selection()
        self.nuscene_samples = []

        # Only load nuScenes for experiments that require real-world data
        if self.cfg.experiment in ('ego-exo-mixed-domain', 'ego-ego-nuscenes', 'ego-exo-nuscenes', 'ego-exo-nuscenes-scene'):
            version = 'v1.0-trainval' if stage in ['train', 'val'] else 'v1.0-test'
            self.nusc = NuScenes(version=version, dataroot=NUSCENE_DATA_DIR)
            all_splits = create_splits_scenes()
            if version == 'v1.0-trainval':
                self.nuscene_scenes = all_splits['train'] if stage == 'train' else all_splits['val']
            else:
                self.nuscene_scenes = all_splits['test']
            
            # Collect all sample tokens for scenes belonging to the current split,
            # by walking each scene's temporal chain from first to last sample
            for scene in self.nusc.scene:
                if scene["name"] in self.nuscene_scenes:
                    sample_token = scene["first_sample_token"]
                    while sample_token:
                        self.nuscene_samples.append(sample_token)
                        sample = self.nusc.get("sample", sample_token)
                        sample_token = sample["next"]
            
            total_samples_before = len(self.nuscene_samples)
            
            if version == 'v1.0-trainval':
                night_scenes_file = '/app/felix/code/Sim2Real/domain_adaptation/nuscene_night_scenes_felix/data/nuscenes_v1.0-trainval_night_scenes.txt'
            else:
                night_scenes_file = '/app/felix/code/Sim2Real/domain_adaptation/nuscene_night_scenes_felix/data/nuscenes_v1.0-test_night_scenes.txt'
            
            with open(night_scenes_file, 'r') as f:
                night_sample_tokens = set(line.strip() for line in f if line.strip())
            
            outlier_poses_file = '/app/felix/code/Sim2Real/camera_setup_comparison/nuscenes_camera_setup/nuscenes_15_outlier_poses.txt'
            with open(outlier_poses_file, 'r') as f:
                outlier_pose_tokens = set(line.strip() for line in f if line.strip())
            
            # Remove night samples and frames with degenerate camera poses to improve
            # domain adaptation quality — only daytime, well-calibrated samples are kept
            tokens_to_filter = night_sample_tokens | outlier_pose_tokens
            self.nuscene_samples = [token for token in self.nuscene_samples if token not in tokens_to_filter]
            
            night_samples_count = len(night_sample_tokens & set(self.nuscene_samples + list(tokens_to_filter)))
            outlier_samples_count = len(outlier_pose_tokens & set(self.nuscene_samples + list(tokens_to_filter)))
            total_filtered = total_samples_before - len(self.nuscene_samples)
            
            print(f"Filtered out the night samples and outlier poses from the nuScenes dataset for domain adaptation: "
                f"{total_samples_before} total scenes, {night_samples_count} night scenes, "
                f"{outlier_samples_count} outlier poses, {total_filtered} total filtered, "
                f"{len(self.nuscene_samples)} scenes left after filtering")
            
            # ~~~ ego-exo-nuscenes-scene ~~~
            # For scene-level evaluation, restrict the dataset to one or more specific nuScenes
            # scenes identified by index within the split, rather than using all samples
            if self.cfg.experiment == 'ego-exo-nuscenes-scene':
                scene_indices = self.cfg.nuscene_scene_index if self.cfg.nuscene_scene_index is not None else [0]
                split_scenes = [s for s in self.nusc.scene if s["name"] in self.nuscene_scenes]

                all_ordered_tokens = []
                self.token_to_scene_name = {}

                for scene_index in scene_indices:
                    assert scene_index < len(split_scenes), (
                        f"nuscene_scene_index={scene_index} is out of range: only {len(split_scenes)} scenes in split")
                    selected_scene = split_scenes[scene_index]

                    token = selected_scene["first_sample_token"]
                    scene_tokens = []
                    while token:
                        scene_tokens.append(token)
                        self.token_to_scene_name[token] = selected_scene["name"]
                        sample = self.nusc.get("sample", token)
                        token = sample["next"]

                    all_ordered_tokens.extend(scene_tokens)
                    print(f"[ego-exo-nuscenes-scene] Selected scene '{selected_scene['name']}' "
                          f"(index {scene_index} in split). "
                          f"Frames: {len(scene_tokens)}")

                self.nuscene_samples = all_ordered_tokens
                print(f"[ego-exo-nuscenes-scene] Total frames across {len(scene_indices)} scene(s): {len(self.nuscene_samples)}")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            print(f"Skipping nuScenes initialization for experiment type: {self.cfg.experiment}")
        
        data_dir_naming = '/ClearNoon/vehicle.audi.tt/'
        
        training_towns = self.cfg.training_towns if self.cfg.training_towns is not None else ['02']
        testing_towns = self.cfg.testing_towns if self.cfg.testing_towns is not None else ['02']
        
        # Build lists of input/output transform file paths for each CARLA spawn point,
        # varying by experiment type (ego-only, ego+exo, mixed-domain, etc.)
        if (self.stage == 'train'):
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns]
            self.spawn_dirs = [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle') for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs)
  
            # ~~~ ego-exo-mixed ~~~
            if self.cfg.experiment == 'ego-exo-mixed':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs
            # ~~~ ego-exo ~~~
            elif self.cfg.experiment == 'ego-exo':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_train.json' for spawn_dir in self.spawn_dirs]
                #self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_BEV70-99_train.json' for spawn_dir in self.spawn_dirs] #BEV modification
            # ~~~ ego-exo-mixed-domain ~~~
            if self.cfg.experiment == 'ego-exo-mixed-domain':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs

        elif (self.stage == 'val'):
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns]
            self.spawn_dirs = [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle') for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs)
            
            # ~~~ ego-exo-mixed ~~~
            if self.cfg.experiment == 'ego-exo-mixed':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs
            # ~~~ ego-exo ~~~
            elif self.cfg.experiment == 'ego-exo':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]
                #self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_BEV70-99_test.json' for spawn_dir in self.spawn_dirs] #BEV modification
            # ~~~ ego-exo-mixed-domain ~~~
            if self.cfg.experiment == 'ego-exo-mixed-domain':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs

        elif (self.stage == 'test'):
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in testing_towns]
            self.spawn_dirs = [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle') for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs)

            # ~~~ ego-exo ~~~
            if self.cfg.experiment == 'ego-exo':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]
                #self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_BEV70-99_test.json' for spawn_dir in self.spawn_dirs] #BEV modification
            # ~~~ ego-ego ~~~
            elif self.cfg.experiment == 'ego-ego':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
            # ~~~ ego-exo-mixed-domain ~~~
            if self.cfg.experiment == 'ego-exo-mixed-domain':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs
            # ~~~ ego-ego-nuscenes ~~~
            if self.cfg.experiment == 'ego-ego-nuscenes':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs
            # ~~~ ego-exo-nuscenes ~~~
            if self.cfg.experiment == 'ego-exo-nuscenes':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                #self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_BEV70-99_test.json' for spawn_dir in self.spawn_dirs] #BEV modification
                self.output_images = self.spawn_dirs
            # ~~~ ego-exo-nuscenes-scene ~~~
            if self.cfg.experiment == 'ego-exo-nuscenes-scene':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs
            
        else:
            raise ValueError("Trying to call dataset class for other purposes is not allowed")
        
        self.input_spawns = np.array(self.input_images)
        self.output_spawns = np.array(self.output_images)
        
        self.context_resolution = (self.view_sampler.cfg.input_context_resolution, self.view_sampler.cfg.input_context_resolution)
        self.target_resolution = (self.view_sampler.cfg.output_target_resolution, self.view_sampler.cfg.output_target_resolution)
        
        self.augment_flag = self.stage == 'train' and self.cfg.train_view_sampler.augment
        
        # Pre-cache all camera data (image paths, intrinsics, extrinsics) at init time to avoid
        # repeated disk reads during training. This populates self.all_texture_context/target etc.
        print("Loading input data...")
        test_input = [self.load_input_example_id(idx) for idx in range(0, self.__len__())]

        print('stage', self.stage)
        print('test_input', test_input)
        print('test_input length', len(test_input))
        
        print("Loading output data...")
        test_output = [self.load_output_example_id(idx) for idx in range(0, self.__len__())]
        print('test_output', test_output)
        print('test_output length', len(test_output))
        
        if self.cfg.experiment in ('ego-ego-nuscenes', 'ego-exo-nuscenes', 'ego-exo-nuscenes-scene'):
            print(f"Carla Dataset, initialized for {self.stage} stage, will use # {len(self.nuscene_samples)} nuScenes samples with augmentation = {self.augment_flag}")
        else:
            print(f"Carla Dataset, initialized for {self.stage} stage, will use # {len(self.input_spawns)} spawns with augmentation = {self.augment_flag}")
        print(f"Training towns: {training_towns}, Testing towns: {testing_towns}")
        print(f"Selected sensors: {self.sensor_indices}")
    
    def _configure_sensor_selection(self) -> List[int]:
        """Configure which sensors to use based on configuration."""
        if self.cfg.selected_sensors is not None:
            sensor_indices = self.cfg.selected_sensors
            print(f"Using explicitly selected sensors: {sensor_indices}")
        else:
            sensor_indices = list(range(6))
            print(f"Using default sensor range: {sensor_indices}")
        
        return sensor_indices
    
    def _filter_context_sensor_data(self, image_paths: List[str], intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> tuple:
        """Filter CONTEXT sensor data based on selected sensor indices. Only applies to ego vehicle sensors."""
        filtered_image_paths = [image_paths[i] for i in self.sensor_indices]
        filtered_intrinsics = intrinsics[self.sensor_indices]
        filtered_extrinsics = extrinsics[self.sensor_indices]
        return filtered_image_paths, filtered_intrinsics, filtered_extrinsics
    
    def _load_nuscene_frame_data(self, sample_token: str):
        """Load nuScenes frame data including images, intrinsics, extrinsics, and sample_data tokens."""
        frame_information = []
        nuscene_frame_info = self.nusc.get("sample", sample_token)
        sample_frame_info = nuscene_frame_info["data"]
        
        for sensor in desired_sensor_names:
            sample_data_token = sample_frame_info[sensor]
            sensor_data = self.nusc.get("sample_data", sample_data_token)
            
            # Retrieve the ego-vehicle global pose at the time this sensor frame was captured
            ego_pose_info = self.nusc.get(table_name="ego_pose", token=sensor_data["ego_pose_token"])
            ego_pose_rotation = Quaternion(ego_pose_info["rotation"])
            ego_pose_translation = np.array(ego_pose_info["translation"])
            
            # Retrieve the sensor's extrinsic calibration (pose relative to the ego vehicle)
            sensor_pose_info = self.nusc.get(table_name="calibrated_sensor",
                                            token=sensor_data["calibrated_sensor_token"])
            sensor_intrinsic = np.array(sensor_pose_info["camera_intrinsic"])
            sensor_pose_rotation = Quaternion(sensor_pose_info["rotation"])
            sensor_pose_translation = np.array(sensor_pose_info["translation"])
            
            # Build the 4x4 sensor-to-world extrinsic matrix using the calibrated sensor pose
            sensor_transform = transform_matrix(sensor_pose_translation, sensor_pose_rotation)
            
            sensor_file_path = NUSCENE_DATA_DIR + sensor_data["filename"]
            
            # Normalize the intrinsic matrix by image dimensions so values are in [0, 1] range,
            # matching the convention used by PixelSplat / SEED4D
            intrinsic_normal = np.zeros((3, 3))
            intrinsic_normal[0, 0] = sensor_intrinsic[0, 0] / sensor_data["width"]
            intrinsic_normal[1, 1] = sensor_intrinsic[1, 1] / sensor_data["height"]
            intrinsic_normal[2, 2] = 1
            intrinsic_normal[0, 2] = sensor_intrinsic[0, 2] / sensor_data["width"]
            intrinsic_normal[1, 2] = sensor_intrinsic[1, 2] / sensor_data["height"]
            
            frame_information.append({
                'image_path': sensor_file_path,
                'intrinsic': torch.from_numpy(intrinsic_normal.astype(np.float32)),
                'extrinsic': torch.from_numpy(sensor_transform.astype(np.float32)),
                'sample_data_token': sample_data_token
            })
        
        return frame_information
    
    def __len__(self):
        if self.cfg.experiment == 'ego-ego-nuscenes':
            return len(self.nuscene_samples)
        elif self.cfg.experiment == 'ego-exo-nuscenes-scene':
            return len(self.nuscene_samples)
        else:
            return len(self.input_spawns)
    
    def get_bound(
        self,
        bound: Literal["z_near", "z_far", "fov"],
        num_views: int) -> Float[Tensor, " view"]:
        if bound == 'z_near':
            value = torch.tensor(self.cfg.z_near, dtype=torch.float32)
        elif bound == 'z_far':
            value = torch.tensor(self.cfg.z_far, dtype=torch.float32)
        elif bound == 'fov':
            value = torch.tensor(self.cfg.max_fov, dtype=torch.float32)
        else:
            raise KeyError("Wrong bound type is passed to retrieve")
        return repeat(value, "-> v", v=num_views)

    def get_input_example_id(self, index):
        """Get example_id for input/context data"""
        if self.cfg.experiment in ('ego-ego-nuscenes', 'ego-exo-nuscenes', 'ego-exo-nuscenes-scene'):
            if index < len(self.nuscene_samples):
                return self.nuscene_samples[index]
            else:
                raise IndexError(f"Index {index} out of bounds for {len(self.nuscene_samples)} nuScenes samples")
        
        intrin_path = self.input_spawns[index]
        example_id = intrin_path[:find_nth_reverse(intrin_path, '/', 3)]
        return example_id

    def get_output_example_id(self, index):
        """Get example_id for output/target data"""
        if self.cfg.experiment == 'ego-ego-nuscenes':
            if index < len(self.nuscene_samples):
                return self.nuscene_samples[index]
            else:
                raise IndexError(f"Index {index} out of bounds for {len(self.nuscene_samples)} nuScenes samples")
        elif self.cfg.experiment == 'ego-exo-nuscenes-scene':
            # All nuScenes frames share the same fixed CARLA spawn as the exo target
            example_id = self.output_spawns[0]
        elif self.cfg.experiment == 'ego-exo-nuscenes':
            example_id = self.output_spawns[index]
        elif self.cfg.experiment in ('ego-exo-mixed', 'ego-exo-mixed-domain'):
            example_id = self.output_spawns[index]
        elif self.cfg.experiment == 'ego-exo' or self.cfg.experiment == 'ego-ego':
            output_intrin_path = self.output_spawns[index]
            example_id = output_intrin_path[:find_nth_reverse(output_intrin_path, '/', 3)]
        else:
            raise ValueError(f"Unknown experiment type: {self.cfg.experiment}")
        
        return example_id

    def _is_nuscene_token(self, example_id):
        """Check if example_id is a nuScenes sample token (32-char hex string)."""
        return (isinstance(example_id, str) and
                len(example_id) == 32 and
                example_id in self.nuscene_samples)
    
    def load_input_example_id(self, index):
        """Load and cache input/context data"""
        example_id = self.get_input_example_id(index)
        
        if not hasattr(self, "all_texture_context"):
            self.all_texture_context = {}
            self.intrinsics_context = {}
            self.extrinsics_context = {}
            
        if example_id not in self.all_texture_context.keys():
            self.all_texture_context[example_id] = []
            self.intrinsics_context[example_id] = []
            self.extrinsics_context[example_id] = []
            
            # nuScenes-only experiments: load all 6 surround cameras directly from the nuScenes API
            if self._is_nuscene_token(example_id):
                frame_information = self._load_nuscene_frame_data(example_id)
                
                for frame_data in frame_information:
                    self.all_texture_context[example_id].append(frame_data['image_path'])
                    self.intrinsics_context[example_id].append(frame_data['intrinsic'])
                    self.extrinsics_context[example_id].append(frame_data['extrinsic'])
                    self.nuscene_path_to_token[frame_data['image_path']] = frame_data['sample_data_token']
                
                self.intrinsics_context[example_id] = torch.stack(self.intrinsics_context[example_id]).cpu()
                self.extrinsics_context[example_id] = torch.stack(self.extrinsics_context[example_id]).cpu()
            
            else:
                # CARLA-based experiments: read camera data from PixelSplat-format transform JSON
                # and restrict to the configured subset of ego-vehicle sensors
                input_transforms = self.input_spawns[index]
                
                context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices = readPixelSplatCamera(
                    input_transforms, resolution=self.view_sampler.cfg.input_context_resolution,
                    near=self.cfg.z_near, far=self.cfg.z_far)
                
                context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices = self._filter_context_sensor_data(
                    context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices)
                
                for image_path, intrins, extrins in zip(context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices):
                    self.all_texture_context[example_id].append(image_path)
                    self.intrinsics_context[example_id].append(intrins)
                    self.extrinsics_context[example_id].append(extrins)
                
                # ~~~ ego-exo-mixed-domain / ego-ego-nuscenes / ego-exo-nuscenes ~~~
                # For mixed-domain experiments, augment the CARLA context with a randomly
                # sampled nuScenes frame so the model sees both synthetic and real data
                # within the same context window
                if self.cfg.experiment in ('ego-exo-mixed-domain',
                               'ego-ego-nuscenes',
                               'ego-exo-nuscenes'):
                    
                    if len(self.nuscene_samples) > 0:
                        nuscene_sample_token = random.choice(self.nuscene_samples)
                        self.nuscene_token_per_example[example_id] = nuscene_sample_token
                        
                        nuscene_frame_data = self._load_nuscene_frame_data(nuscene_sample_token)
                        
                        if nuscene_frame_data:
                            print(f"[DEBUG LOAD_INPUT] Sample nuScenes context image path: {nuscene_frame_data[0]['image_path']}")
                        
                        nuscene_context_count = 0
                        sample_data_tokens_for_example = []
                        for frame_data in nuscene_frame_data:
                            self.all_texture_context[example_id].append(frame_data['image_path'])
                            self.intrinsics_context[example_id].append(frame_data['intrinsic'])
                            self.extrinsics_context[example_id].append(frame_data['extrinsic'])
                            sample_data_tokens_for_example.append(frame_data['sample_data_token'])
                            nuscene_context_count += 1

                        self.nuscene_sample_data_tokens_per_example[example_id] = sample_data_tokens_for_example
                        
                        for frame_data in nuscene_frame_data:
                            self.nuscene_path_to_token[frame_data['image_path']] = frame_data['sample_data_token']
                    else:
                        print(f"[DEBUG LOAD_INPUT WARNING] No nuScenes samples available for stage {self.stage}")
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                
                self.intrinsics_context[example_id] = torch.stack(self.intrinsics_context[example_id]).cpu()
                self.extrinsics_context[example_id] = torch.stack(self.extrinsics_context[example_id]).cpu()
        
        return None

    def load_output_example_id(self, index):
        """Load and cache output/target data"""
        example_id = self.get_output_example_id(index)
        
        if not hasattr(self, "all_texture_target"):
            self.all_texture_target = {}
            self.intrinsics_target = {}
            self.extrinsics_target = {}
            
        if example_id not in self.all_texture_target.keys():
            self.all_texture_target[example_id] = []
            self.intrinsics_target[example_id] = []
            self.extrinsics_target[example_id] = []
            
            if self._is_nuscene_token(example_id):
                frame_information = self._load_nuscene_frame_data(example_id)
                
                for frame_data in frame_information:
                    self.all_texture_target[example_id].append(frame_data['image_path'])
                    self.intrinsics_target[example_id].append(frame_data['intrinsic'])
                    self.extrinsics_target[example_id].append(frame_data['extrinsic'])
                    self.nuscene_path_to_token[frame_data['image_path']] = frame_data['sample_data_token']
                
                self.intrinsics_target[example_id] = torch.stack(self.intrinsics_target[example_id]).cpu()
                self.extrinsics_target[example_id] = torch.stack(self.extrinsics_target[example_id]).cpu()
            
            elif self.stage == 'train' or self.stage == 'val' or self.stage == 'test':
                # ~~~ ego-exo-mixed ~~~
                if self.cfg.experiment == 'ego-exo-mixed':
                    exo_transforms = example_id + '/sphere_invisible/transforms/transforms_ego_train.json'
                    exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices = readPixelSplatCamera(
                        exo_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    print(f"Stage {self.stage}: Loading {len(exo_image_paths)} EXO target views from {exo_transforms}")
                    
                    for image_path, intrins, extrins in zip(exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)
                    
                    ego_transforms = example_id + '/nuscenes_invisible/transforms/transforms_ego.json'
                    ego_image_paths, ego_intrinsics_matrices, ego_extrinsics_matrices = readPixelSplatCamera(
                        ego_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    filtered_ego_image_paths = [ego_image_paths[i] for i in self.sensor_indices]
                    filtered_ego_intrinsics = ego_intrinsics_matrices[self.sensor_indices]
                    filtered_ego_extrinsics = ego_extrinsics_matrices[self.sensor_indices]

                    print(f"Stage {self.stage}: Filtered to {len(filtered_ego_image_paths)} EGO views using sensor indices {self.sensor_indices}")

                    # Repeat ego views 3x to balance their count against the larger set of exo views
                    for _ in range(3):
                        for image_path, intrins, extrins in zip(filtered_ego_image_paths, filtered_ego_intrinsics, filtered_ego_extrinsics):
                            self.all_texture_target[example_id].append(image_path)
                            self.intrinsics_target[example_id].append(intrins)
                            self.extrinsics_target[example_id].append(extrins)

                # ~~~ ego-exo-nuscenes ~~~
                elif self.cfg.experiment == 'ego-exo-nuscenes':
                    exo_transforms = example_id + '/sphere_invisible/transforms/transforms_ego_train.json'
                    #exo_transforms = example_id + '/sphere_invisible/transforms/transforms_ego_BEV70-99_test.json' #changes for BEV
                    exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices = readPixelSplatCamera(
                        exo_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    print(f"Stage {self.stage}: Loading {len(exo_image_paths)} EXO target views (sphere) for {self.cfg.experiment}")
                    
                    for image_path, intrins, extrins in zip(exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)

                # ~~~ ego-exo-nuscenes-scene ~~~
                elif self.cfg.experiment == 'ego-exo-nuscenes-scene':
                    exo_transforms = example_id + '/sphere_invisible/transforms/transforms_ego.json'
                    exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices = readPixelSplatCamera(
                        exo_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    print(f"Stage {self.stage}: Loading {len(exo_image_paths)} EXO target views (sphere) for {self.cfg.experiment}")
                    
                    for image_path, intrins, extrins in zip(exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)

                # ~~~ ego-exo-mixed-domain / ego-ego-nuscenes ~~~
                elif self.cfg.experiment in ('ego-exo-mixed-domain',
                           'ego-ego-nuscenes'):
                    exo_transforms = example_id + '/sphere_invisible/transforms/transforms_ego_train.json'
                    exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices = readPixelSplatCamera(
                        exo_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    print(f"Stage {self.stage}: Loading {len(exo_image_paths)} EXO target views from {exo_transforms}")
                    
                    for image_path, intrins, extrins in zip(exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)
                    
                    ego_transforms = example_id + '/nuscenes_invisible/transforms/transforms_ego.json'
                    ego_image_paths, ego_intrinsics_matrices, ego_extrinsics_matrices = readPixelSplatCamera(
                        ego_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    filtered_ego_image_paths = [ego_image_paths[i] for i in self.sensor_indices]
                    filtered_ego_intrinsics = ego_intrinsics_matrices[self.sensor_indices]
                    filtered_ego_extrinsics = ego_extrinsics_matrices[self.sensor_indices]

                    print(f"Stage {self.stage}: Filtered to {len(filtered_ego_image_paths)} EGO views using sensor indices {self.sensor_indices}")

                    # Repeat ego views 3x to balance their count against the larger set of exo views
                    for _ in range(3):
                        for image_path, intrins, extrins in zip(filtered_ego_image_paths, filtered_ego_intrinsics, filtered_ego_extrinsics):
                            self.all_texture_target[example_id].append(image_path)
                            self.intrinsics_target[example_id].append(intrins)
                            self.extrinsics_target[example_id].append(extrins)

                    # Append nuScenes target views using the same sample token that was paired
                    # with this CARLA example during input loading, ensuring context/target alignment
                    if len(self.nuscene_samples) > 0:
                        assert example_id in self.nuscene_token_per_example, \
                            f"Token should exist for {example_id} - load_input_example_id should run first"
                        nuscene_sample_token = self.nuscene_token_per_example[example_id]
                        
                        nuscene_frame_data = self._load_nuscene_frame_data(nuscene_sample_token)
                        filtered_nuscene_data = nuscene_frame_data
                        print(f"Stage {self.stage}: Loading {len(filtered_nuscene_data)} nuScenes target views")
                        
                        for frame_data in filtered_nuscene_data:
                            self.all_texture_target[example_id].append(frame_data['image_path'])
                            self.intrinsics_target[example_id].append(frame_data['intrinsic'])
                            self.extrinsics_target[example_id].append(frame_data['extrinsic'])

                # ~~~ ego-exo ~~~
                elif self.cfg.experiment == 'ego-exo':
                    output_transforms = self.output_spawns[index]
                    print(f"[DEBUG ego-exo] output_transforms = {output_transforms}")
                    target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = readPixelSplatCamera(
                        output_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                
                    print(f"Stage {self.stage}: Loading {len(target_image_paths)} target views from {output_transforms}")
                
                    filtered_target_image_paths = target_image_paths
                    filtered_target_intrinsics = target_intrinsics_matrices
                    filtered_target_extrinsics = target_extrinsics_matrices
                    
                    for image_path, intrins, extrins in zip(filtered_target_image_paths, filtered_target_intrinsics, filtered_target_extrinsics):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)

                # ~~~ ego-ego ~~~
                elif self.cfg.experiment == 'ego-ego':
                    output_transforms = self.output_spawns[index]
                    target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = readPixelSplatCamera(
                        output_transforms, resolution=self.view_sampler.cfg.output_target_resolution,
                        near=self.cfg.z_near, far=self.cfg.z_far)
                
                    print(f"Stage {self.stage}: Loading {len(target_image_paths)} target views from {output_transforms}")
                
                    filtered_target_image_paths = [target_image_paths[i] for i in self.sensor_indices]
                    filtered_target_intrinsics = target_intrinsics_matrices[self.sensor_indices]
                    filtered_target_extrinsics = target_extrinsics_matrices[self.sensor_indices]
                    
                    print(f"Stage {self.stage}: Filtered to {len(filtered_target_image_paths)} target views using sensor indices {self.sensor_indices}")
                    
                    for image_path, intrins, extrins in zip(filtered_target_image_paths, filtered_target_intrinsics, filtered_target_extrinsics):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)
                
                self.intrinsics_target[example_id] = torch.stack(self.intrinsics_target[example_id]).cpu()
                self.extrinsics_target[example_id] = torch.stack(self.extrinsics_target[example_id]).cpu()
            else:
                assert False, "Experiment is not set properly"
        
        return None

    def __getitem__(self, index):
        input_example_id = self.get_input_example_id(index)
        output_example_id = self.get_output_example_id(index)

        # Determine whether this batch item should use nuScenes data for the context/target.
        # For mixed-domain training, only 1 in 50 samples switches to nuScenes to maintain
        # a predominantly synthetic training distribution
        if self.cfg.experiment in ('ego-ego-nuscenes', 'ego-exo-nuscenes', 'ego-exo-nuscenes-scene'):
            use_nuscene_for_this_sample = True
        elif self.cfg.experiment == 'ego-exo-mixed-domain':
            use_nuscene_for_this_sample = (index % 50 == 0)
        else:
            use_nuscene_for_this_sample = False

        index_context, index_target = self.view_sampler.sample("SEED",
                                                               self.extrinsics_context[input_example_id],
                                                               self.extrinsics_target[output_example_id],
                                                               experiment=self.cfg.experiment,
                                                               use_nuscene_context=use_nuscene_for_this_sample)
        
        context_images = [img_path_to_Torch(image_path, self.context_resolution)
                          for image_path in np.array(self.all_texture_context[input_example_id])[index_context.numpy()]]
        context_images = torch.stack(context_images).float()
        context_extrinsics = self.extrinsics_context[input_example_id][index_context.numpy()]
        context_intrinsics = self.intrinsics_context[input_example_id][index_context.numpy()]
        
        # Guard against view sampler returning indices beyond the cached target view count,
        # which can happen when nuScenes and CARLA targets differ in cardinality
        max_available_views = len(self.all_texture_target[output_example_id])
        if any(idx >= max_available_views for idx in index_target.numpy()):
            print(f"Warning: Requested indices {index_target.numpy()} exceed available views ({max_available_views}) for {output_example_id}")
            index_target = torch.clamp(index_target, 0, max_available_views - 1)
            print(f"Clamped indices to: {index_target.numpy()}")
        
        target_images = [img_path_to_Torch(image_path, self.target_resolution)
                         for image_path in np.array(self.all_texture_target[output_example_id])[index_target.numpy()]]
        target_images = torch.stack(target_images).float()
        
        # Load depth maps for each target view. nuScenes targets use precomputed Depth-Anything-3
        # .npy files (scaled by 1/100 to convert cm→m); SEED4D targets use paired _depth.png files
        target_depths = []
        for image_path in np.array(self.all_texture_target[output_example_id])[index_target.numpy()]:
            if '/samples/' in image_path and 'nuscenes_full' in image_path:
                if image_path in self.nuscene_path_to_token:
                    sample_data_token = self.nuscene_path_to_token[image_path]
                    
                    if self.stage == 'train' or self.stage == 'val':
                        depth_file_path = f'/app/felix/data/depth_anything3/nuscenes_depth_trainval_800/{sample_data_token}_depth.npy'
                    else:
                        depth_file_path = f'/app/felix/data/depth_anything3/nuscenes_depth_test_800/{sample_data_token}_depth.npy'
                    
                    if os.path.exists(depth_file_path):
                        depth_npy = np.load(depth_file_path)
                        depth_meters = depth_npy.astype(np.float32) / 100.0
                        depth_tensor = torch.from_numpy(depth_meters).float()
                        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
                        depth_upscaled = F.interpolate(depth_tensor, size=self.target_resolution, mode='nearest')
                        depth = depth_upscaled.squeeze(0).squeeze(0)
                    else:
                        depth = torch.zeros(self.target_resolution, dtype=torch.float32)
                else:
                    depth = torch.zeros(self.target_resolution, dtype=torch.float32)
            else:
                depth_path = image_path[:image_path.rfind('_')] + "_depth.png"
                if os.path.exists(depth_path):
                    depth = depth_path_to_Torch(depth_path, self.target_resolution)
                else:
                    print(f"Warning: Depth map not found for SEED4D image {image_path}")
                    depth = torch.zeros(self.target_resolution, dtype=torch.float32)
            
            target_depths.append(depth)
        
        target_depths = torch.stack(target_depths).float()

        target_extrinsics = self.extrinsics_target[output_example_id][index_target.numpy()]
        target_intrinsics = self.intrinsics_target[output_example_id][index_target.numpy()]
        
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
                    "scene": (self.token_to_scene_name[input_example_id]
                               if self.cfg.experiment == 'ego-exo-nuscenes-scene'
                               and hasattr(self, 'token_to_scene_name')
                               and input_example_id in self.token_to_scene_name
                               else "Carla"),
                    "dataset_change": use_nuscene_for_this_sample}

        return example
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ File and directory utilities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def str_list_concat(pre_string, folders_dir, post_string):
    list_dirs = os.listdir(folders_dir)
    return [pre_string + directory + '/' + post_string for directory in list_dirs]

def find_nth_reverse(haystack: str, needle: str, n: int) -> int:
    # copied from https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string
    end = haystack.rfind(needle)
    while end >= 0 and n > 1:
        end = haystack.rfind(needle, 0, end - len(haystack))
        n -= 1
    return end

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
    """Apply rotation transformation to match the coordinate system change."""
    rotation_matrix = quaternion.rotation_matrix
    
    # Transformation matrix T aligns nuScenes/CARLA axes to PixelSplat's expected coordinate frame
    T = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0, -1,  0]
    ])
    
    transformed_rotation_matrix = T @ rotation_matrix
    transformed_quaternion = Quaternion(matrix=transformed_rotation_matrix)
    
    return transformed_quaternion

def quaternion_to_transform_matrix(quaternion, translation):
    """Convert quaternion and translation to 4x4 transformation matrix."""
    rotation_matrix = quaternion.rotation_matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    New transformation to match the format pixelsplat was trained on (format of seed4d/carla generator).
    """
    # Flip the x-axis to convert from nuScenes (right-handed, z-up) to PixelSplat's camera convention
    x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
    original_quaternion = rotation * x_axis_flip
        
    transformed_translation = apply_coordinate_transformation(translation)
    transformed_quaternion = apply_rotation_transformation(original_quaternion)
        
    transformed_transform_matrix = quaternion_to_transform_matrix(
        transformed_quaternion, transformed_translation
    )

    # Negate columns 1 and 2 of the rotation block to finalize the camera-convention alignment
    # (equivalent to flipping the y and z axes of the camera frame)
    transformed_transform_matrix[0, 1] *= -1
    transformed_transform_matrix[0, 2] *= -1
    transformed_transform_matrix[1, 1] *= -1
    transformed_transform_matrix[1, 2] *= -1
    transformed_transform_matrix[2, 1] *= -1
    transformed_transform_matrix[2, 2] *= -1

    return transformed_transform_matrix