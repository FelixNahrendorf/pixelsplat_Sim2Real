#This file is used for the sensitivity analysis of the SEED4D dataset regarding the camera parameters

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
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
# from .shims.augmentation_shim import apply_augmentation_shim
# from .shims.mask_shim import apply_mask_shim
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerCfg

from .dataset_readers import readPixelSplatCamera
from ..misc.general_utils import img_path_to_Torch, depth_path_to_Torch

SEED4D_DATASET_ROOT = '/app/inputs/seed4d/data/data_diverse/static/'  #'/app/inputs/seed4d/data/data_baseline/static/' 
assert SEED4D_DATASET_ROOT is not None, "Update the location of the SEED4D Dataset"

LIDAR_DATASET_ROOT = '/app/new/seed4d/pseudo_lidar/' # Will be directory to save pseudo lidar 

@dataclass
class Dataset_SEED4DCfg(DatasetCfgCommon):
    name: Literal["seed4d"]
    train_view_sampler: ViewSamplerCfg
    eval_view_sampler: ViewSamplerCfg
    max_fov: float
    z_near: float
    z_far: float
    experiment: str
    training_towns: List[str] = None  # Add training towns configuration
    testing_towns: List[str] = None   # Add testing towns configuration
    selected_sensors: Optional[List[int]] = None  # List of sensor indices to use
    #sensor_range: Optional[List[int]] = None  # Alternative: [start, end] range of sensors
    

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
        
        # Configure sensor selection
        self.sensor_indices = self._configure_sensor_selection()
        
        data_dir_naming = '/ClearNoon/vehicle.audi.tt/'
        
        training_towns = self.cfg.training_towns if self.cfg.training_towns is not None else ['02']
        testing_towns = self.cfg.testing_towns if self.cfg.testing_towns is not None else ['02']
        
        if (self.stage == 'train'):
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns]
            self.spawn_dirs = [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle') for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
  
            ### ego-exo-mixed training
            assert self.cfg.experiment is not None
            if self.cfg.experiment == 'ego-exo-mixed':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs 
            ###ego-exo training 
            elif self.cfg.experiment == 'ego-exo':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_train.json' for spawn_dir in self.spawn_dirs]

        elif (self.stage == 'val'): 
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns]
            self.spawn_dirs = [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle') for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
            
            ### ego-exo-mixed training
            assert self.cfg.experiment is not None
            if self.cfg.experiment == 'ego-exo-mixed':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = self.spawn_dirs  
            ###ego-exo training 
            elif self.cfg.experiment == 'ego-exo':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]

        elif (self.stage == 'test'): 
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in testing_towns]
            self.spawn_dirs = [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle') for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            print(f"DEBUG TEST STAGE: Total spawn directories found: {len(self.spawn_dirs)}")
            random.shuffle(self.spawn_dirs) 

            assert self.cfg.experiment is not None
            ### ego-exo testing
            if self.cfg.experiment == 'ego-exo':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]
            ### ego-ego testing
            elif self.cfg.experiment == 'ego-ego':
                self.input_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
                self.output_images = [spawn_dir + '/nuscenes_invisible/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
            
        else: raise ValueError("Trying to call dataset class for other purposes is not allowed")
        
        self.input_spawns = np.array(self.input_images)
        self.output_spawns = np.array(self.output_images)
        
        # # Selected randomly to train just on a single spawn point
        #self.input_spawns = np.array(self.input_images)[:10]
        #self.output_spawns = np.array(self.output_images)[:10]
        
        # configuring relevant resolution for inward and outward facing cameras
        self.context_resolution = (self.view_sampler.cfg.input_context_resolution, self.view_sampler.cfg.input_context_resolution)
        self.target_resolution = (self.view_sampler.cfg.output_target_resolution, self.view_sampler.cfg.output_target_resolution)
        
        #######################################################################
        # Below is the important flag changing workflow of several blocks
        self.augment_flag = self.stage == 'train' and self.cfg.train_view_sampler.augment
        
        # Preload ALL input data (context)
        print("Loading input data...")
        test_input = [self.load_input_example_id(idx) for idx in range(0, len(self.input_spawns))]

        print('stage', self.stage)
        print('test_input', test_input)
        print('test_input length', len(test_input))
        
        # Preload ALL output data (target)  
        print("Loading output data...")
        test_output = [self.load_output_example_id(idx) for idx in range(0, len(self.output_spawns))]
        print('test_output', test_output)
        print('test_output length', len(test_output))
        
        print(f"Carla Dataset, initialized for {self.stage} stage, will use # {len(self.input_spawns)} spawns with augmentation = {self.augment_flag}")
        print(f"Training towns: {training_towns}, Testing towns: {testing_towns}")
        print(f"Selected sensors: {self.sensor_indices}")
    
    def _configure_sensor_selection(self) -> List[int]:
        """Configure which sensors to use based on configuration."""
        if self.cfg.selected_sensors is not None:
            # Use explicitly specified sensor list
            sensor_indices = self.cfg.selected_sensors
            print(f"Using explicitly selected sensors: {sensor_indices}")
        elif hasattr(self.cfg, 'sensor_range') and self.cfg.sensor_range is not None:
            # Use sensor range [start, end] (inclusive)
            start, end = self.cfg.sensor_range
            sensor_indices = list(range(start, end + 1))
            print(f"Using sensor range {start}-{end}: {sensor_indices}")
        else:
            # Default: use all sensors (0-6 based on the JSON structure)
            sensor_indices = list(range(7))  # 0, 1, 2, 3, 4, 5, 6
            print(f"Using default sensors (all): {sensor_indices}")
        
        return sensor_indices
    
    def _filter_context_sensor_data(self, image_paths: List[str], intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> tuple:
        """Filter CONTEXT sensor data based on selected sensor indices. Only applies to ego vehicle sensors."""
        '''print(f'Filtering context sensors - Original count: {len(image_paths)}')
        print(f'Selected sensor indices: {self.sensor_indices}')'''
        
        # Filter image paths using the sensor indices directly
        filtered_image_paths = [image_paths[i] for i in self.sensor_indices]
        
        # Filter tensors using tensor indexing with the sensor indices
        filtered_intrinsics = intrinsics[self.sensor_indices]
        filtered_extrinsics = extrinsics[self.sensor_indices]
        
        '''print(f'Filtering context sensors - Filtered count: {len(filtered_image_paths)}')
        print(f'Filtered intrinsics shape: {filtered_intrinsics.shape}')
        print(f'Filtered extrinsics shape: {filtered_extrinsics.shape}')'''
        
        return filtered_image_paths, filtered_intrinsics, filtered_extrinsics
    
    def __len__(self):
        return len(self.input_spawns)
    
    def get_bound(
        self,
        bound: Literal["z_near", "z_far", "fov"],
        num_views: int) -> Float[Tensor, " view"]:
        # return near and far bounds with shape = (num_views,) 
        if bound=='z_near': value = torch.tensor(self.cfg.z_near, dtype=torch.float32) 
        elif bound=='z_far': value = torch.tensor(self.cfg.z_far, dtype=torch.float32) 
        elif bound=='fov': value = torch.tensor(self.cfg.max_fov, dtype=torch.float32) 
        
        else: raise KeyError("Wrong bound type is passed to retrieve")
        return repeat(value, "-> v", v=num_views)

    def get_input_example_id(self, index):
        """Get example_id for input/context data"""
        intrin_path = self.input_spawns[index]
        example_id = intrin_path[:find_nth_reverse(intrin_path, '/', 3)]
        return example_id

    def get_output_example_id(self, index):
        """Get example_id for output/target data"""
        if self.cfg.experiment == 'ego-exo-mixed':
            example_id = self.output_spawns[index]
        elif self.cfg.experiment == 'ego-exo' or self.cfg.experiment == 'ego-ego':
            output_path = self.output_spawns[index]
            example_id = output_path[:find_nth_reverse(output_path, '/', 3)]
        return example_id
    
    def load_input_example_id(self, index):
        """Load and cache input/context data"""
        example_id = self.get_input_example_id(index)
        input_transforms = self.input_spawns[index]
        
        if not hasattr(self, "all_texture_context"):
            self.all_texture_context = {}
            self.intrinsics_context = {}
            self.extrinsics_context = {}
            
        if example_id not in self.all_texture_context.keys():
            self.all_texture_context[example_id] = []
            self.intrinsics_context[example_id] = []
            self.extrinsics_context[example_id] = []
            
            # Load context data
            context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices = readPixelSplatCamera(
                input_transforms, resolution=self.view_sampler.cfg.input_context_resolution, 
                near=self.cfg.z_near, far=self.cfg.z_far)
            
            # Filter context sensor data
            context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices = self._filter_context_sensor_data(
                context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices)
            
            # Store context data
            for image_path, intrins, extrins in zip(context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices):
                self.all_texture_context[example_id].append(image_path)
                self.intrinsics_context[example_id].append(intrins)
                self.extrinsics_context[example_id].append(extrins)
            
            self.intrinsics_context[example_id] = torch.stack(self.intrinsics_context[example_id]).cpu()
            self.extrinsics_context[example_id] = torch.stack(self.extrinsics_context[example_id]).cpu()
        
        return None  # Just for the list comprehension

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
            
            if self.stage == 'train' or self.stage == 'val' or self.stage == 'test':
                ###ego-ego/ego-exo mixed training only
                if self.cfg.experiment == 'ego-exo-mixed':
                #if os.path.isdir(example_id): ##can be deleted if its running
                    # For training stage, load BOTH exo and ego views for each spawn point
                    
                    # Load exo views (sphere_invisible)
                    exo_transforms = example_id + '/sphere_invisible/transforms/transforms_ego_train.json'
                    exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices = readPixelSplatCamera(
                        exo_transforms, resolution=self.view_sampler.cfg.output_target_resolution, 
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    print(f"Stage {self.stage}: Loading {len(exo_image_paths)} EXO target views from {exo_transforms}")
                    
                    # Store exo data
                    for image_path, intrins, extrins in zip(exo_image_paths, exo_intrinsics_matrices, exo_extrinsics_matrices):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)
                    
                    # Load ego views (nuscenes_invisible)
                    ego_transforms = example_id + '/nuscenes_invisible/transforms/transforms_ego.json'
                    ego_image_paths, ego_intrinsics_matrices, ego_extrinsics_matrices = readPixelSplatCamera(
                        ego_transforms, resolution=self.view_sampler.cfg.output_target_resolution, 
                        near=self.cfg.z_near, far=self.cfg.z_far)
                    
                    # Filter ego sensor data using selected sensor indices
                    filtered_ego_image_paths = [ego_image_paths[i] for i in self.sensor_indices] #uncommented since only sensors 0-5 are generated
                    filtered_ego_intrinsics = ego_intrinsics_matrices[self.sensor_indices] #uncommented since only sensors 0-5 are generated
                    filtered_ego_extrinsics = ego_extrinsics_matrices[self.sensor_indices] #uncommented since only sensors 0-5 are generated

                    print(f"Stage {self.stage}: Filtered to {len(filtered_ego_image_paths)} EGO views using sensor indices {self.sensor_indices}")

                    # Store filtered ego data 
                    for _ in range(3): # Repeat 3 times to balance ego and exo views
                        for image_path, intrins, extrins in zip(filtered_ego_image_paths, filtered_ego_intrinsics, filtered_ego_extrinsics):
                            self.all_texture_target[example_id].append(image_path)
                            self.intrinsics_target[example_id].append(intrins)
                            self.extrinsics_target[example_id].append(extrins)

                
                ###ego-exo training only
                elif self.cfg.experiment == 'ego-exo':
                    output_transforms = self.output_spawns[index]
                    target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = readPixelSplatCamera(
                    output_transforms, resolution=self.view_sampler.cfg.output_target_resolution, 
                    near=self.cfg.z_near, far=self.cfg.z_far)
                
                    print(f"Stage {self.stage}: Loading {len(target_image_paths)} target views from {output_transforms}")
                
                    ### no filtering happens here, can be deleted
                    # Filter target sensor data using selected sensor indices, only when testing ego-ego generation
                    filtered_target_image_paths = target_image_paths #[target_image_paths[i] for i in self.sensor_indices]
                    filtered_target_intrinsics = target_intrinsics_matrices #target_intrinsics_matrices[self.sensor_indices]
                    filtered_target_extrinsics = target_extrinsics_matrices #target_extrinsics_matrices[self.sensor_indices]
                    
                    #print(f"Stage {self.stage}: Filtered to {len(filtered_target_image_paths)} target views using sensor indices {self.sensor_indices}")
                    
                    # Store filtered target data
                    for image_path, intrins, extrins in zip(filtered_target_image_paths, filtered_target_intrinsics, filtered_target_extrinsics):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins)   
                
                elif self.cfg.experiment == 'ego-ego': 
                    output_transforms = self.output_spawns[index]
                    target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = readPixelSplatCamera(
                    output_transforms, resolution=self.view_sampler.cfg.output_target_resolution, 
                    near=self.cfg.z_near, far=self.cfg.z_far)
                
                    print(f"Stage {self.stage}: Loading {len(target_image_paths)} target views from {output_transforms}")
                
                    # Filter target sensor data using selected sensor indices, only when testing ego-ego generation
                    filtered_target_image_paths = [target_image_paths[i] for i in self.sensor_indices]
                    filtered_target_intrinsics = target_intrinsics_matrices[self.sensor_indices]
                    filtered_target_extrinsics = target_extrinsics_matrices[self.sensor_indices]
                    
                    print(f"Stage {self.stage}: Filtered to {len(filtered_target_image_paths)} target views using sensor indices {self.sensor_indices}")
                    
                    # Store filtered target data
                    for image_path, intrins, extrins in zip(filtered_target_image_paths, filtered_target_intrinsics, filtered_target_extrinsics):
                        self.all_texture_target[example_id].append(image_path)
                        self.intrinsics_target[example_id].append(intrins)
                        self.extrinsics_target[example_id].append(extrins) 
            else:
                # For test stages, use the original logic
                output_transforms = self.output_spawns[index]
                target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = readPixelSplatCamera(
                    output_transforms, resolution=self.view_sampler.cfg.output_target_resolution, 
                    near=self.cfg.z_near, far=self.cfg.z_far)
                
                print(f"Stage {self.stage}: Loading {len(target_image_paths)} target views from {output_transforms}")
                
                # Filter target sensor data using selected sensor indices, only when testing ego-ego generation
                filtered_target_image_paths = [target_image_paths[i] for i in self.sensor_indices]
                filtered_target_intrinsics = target_intrinsics_matrices[self.sensor_indices]
                filtered_target_extrinsics = target_extrinsics_matrices[self.sensor_indices]
                
                print(f"Stage {self.stage}: Filtered to {len(filtered_target_image_paths)} target views using sensor indices {self.sensor_indices}")
                
                # Store filtered target data
                for image_path, intrins, extrins in zip(filtered_target_image_paths, filtered_target_intrinsics, filtered_target_extrinsics):
                    self.all_texture_target[example_id].append(image_path)
                    self.intrinsics_target[example_id].append(intrins)
                    self.extrinsics_target[example_id].append(extrins)
            
            self.intrinsics_target[example_id] = torch.stack(self.intrinsics_target[example_id]).cpu()
            self.extrinsics_target[example_id] = torch.stack(self.extrinsics_target[example_id]).cpu()
        
        return None  # Just for the list comprehension
    
    def __getitem__(self, index):
        input_example_id = self.get_input_example_id(index)
        output_example_id = self.get_output_example_id(index)
        
        # view sampler needs to know context and target view extrinsics for sampling strategy
        index_context, index_target = self.view_sampler.sample("SEED", 
                                                               self.extrinsics_context[input_example_id], 
                                                               self.extrinsics_target[output_example_id],
                                                               self.cfg.experiment)
        
        #######################################################################
        #
        #######################################################################
        ############### Loading Context/Conditional Information ###############
        context_images = [img_path_to_Torch(image_path, self.context_resolution)
                          for image_path in np.array(self.all_texture_context[input_example_id])[index_context.numpy()]]
        context_images = torch.stack(context_images).float()
        context_extrinsics = self.extrinsics_context[input_example_id][index_context.numpy()]
        context_intrinsics = self.intrinsics_context[input_example_id][index_context.numpy()]
        
        #######################################################################
        ################ Loading Inference Target Information #################
        
        # Add bounds checking before accessing target images
        max_available_views = len(self.all_texture_target[output_example_id])
        if any(idx >= max_available_views for idx in index_target.numpy()):
            print(f"Warning: Requested indices {index_target.numpy()} exceed available views ({max_available_views}) for {output_example_id}")
            # Clamp indices to valid range
            index_target = torch.clamp(index_target, 0, max_available_views - 1)
            print(f"Clamped indices to: {index_target.numpy()}")
        
        # Reading images
        print('index_target.numpy()', index_target.numpy())
        print('output_example_id', output_example_id)
        print('len self.all_texture_target[output_example_id]', len(self.all_texture_target[output_example_id]))

        target_images = [img_path_to_Torch(image_path, self.target_resolution)
                         for image_path in np.array(self.all_texture_target[output_example_id])[index_target.numpy()]]
        target_images = torch.stack(target_images).float()
        # Reading depth maps
        target_depths = [depth_path_to_Torch(image_path[:image_path.rfind('_')] + "_depth.png", self.target_resolution)
                         for image_path in np.array(self.all_texture_target[output_example_id])[index_target.numpy()]]
        target_depths = torch.stack(target_depths).float()
        # Reading Camera params
        target_extrinsics = self.extrinsics_target[output_example_id][index_target.numpy()]
        target_intrinsics = self.intrinsics_target[output_example_id][index_target.numpy()]
        
        #######################################################################
        #######################################################################
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
                    "scene": "Carla"}


        '''print("=== FINAL DATA FED TO MODEL ===")
        print('context_resolution:', self.context_resolution) ###DEBUG
        print('context_resolution[0]:', self.context_resolution[0]) ###DEBUG
        print(f"Resolution: {self.view_sampler.cfg.output_target_resolution}")
        print(f"Index target: {index_target.numpy()}")
        print(f"Target intrinsics shape: {self.intrinsics_target[output_example_id].shape}")

        print("Context intrinsics shape:", example['context']['intrinsics'].shape)
        print("Context intrinsics:\n", example['context']['intrinsics'])
        print("Context extrinsics shape:", example['context']['extrinsics'].shape)
        print("Context extrinsics:\n", example['context']['extrinsics'])
        print("Target intrinsics shape:", example['target']['intrinsics'].shape) 
        print("Target intrinsics:\n", example['target']['intrinsics'])
        print("Target extrinsics shape:", example['target']['extrinsics'].shape)
        print("Target extrinsics:\n", example['target']['extrinsics'])
        print("="*50)  '''       
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
        end = haystack.rfind(needle, 0, end - len(haystack))
        n -= 1
    return end