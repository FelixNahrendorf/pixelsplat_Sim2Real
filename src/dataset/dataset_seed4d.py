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
from typing import Literal, List
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

SEED4D_DATASET_ROOT = '/app/data/seed4d/static/' # Change this to your data directory 
assert SEED4D_DATASET_ROOT is None, "Update the location of the SEED4D Dataset"

LIDAR_DATASET_ROOT = '/app/new/seed4d/pseudo_lidar/' # Will be directory to save pseudo lidar 

@dataclass
class Dataset_CARLACfg(DatasetCfgCommon):
    name: Literal["seed4d"]
    train_view_sampler: ViewSamplerCfg
    eval_view_sampler: ViewSamplerCfg
    max_fov: float
    z_near: float
    z_far: float
    training_towns: List[str] = None  # Add training towns configuration
    testing_towns: List[str] = None   # Add testing towns configuration

class Dataset_CARLA(Dataset):
    cfg: Dataset_CARLACfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: Dataset_CARLACfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        data_dir_naming = '/ClearNoon/vehicle.audi.tt/'
        
        # Use configuration values or default fallback
        training_towns = self.cfg.training_towns if self.cfg.training_towns is not None else ['02']
        testing_towns = self.cfg.testing_towns if self.cfg.testing_towns is not None else ['02']
        
        if (self.stage == 'train'):
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns] #data/seed4d/static/Town02/ClearNoon
            self.spawn_dirs =  [str_list_concat(spawns_dir, spawns_dir, '/step_0/ego_vehicle')  for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
            self.input_images = [spawn_dir + '/nuscenes/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
            self.output_images = [spawn_dir + '/sphere/transforms/transforms_ego_train.json' for spawn_dir in self.spawn_dirs]
            
        elif (self.stage == 'val'): # val stands for validation
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns]
            self.spawn_dirs =  [str_list_concat(spawns_dir, spawns_dir, '/step_0/ego_vehicle')  for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
            self.input_images = [spawn_dir + '/nuscenes/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
            self.output_images = [spawn_dir + '/sphere/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]
            
        elif (self.stage == 'test'): # val stands for validation
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in testing_towns]
            self.spawn_dirs =  [str_list_concat(spawns_dir, spawns_dir, '/step_0/ego_vehicle')  for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
            self.input_images = [spawn_dir + '/nuscenes/transforms/transforms_ego.json' for spawn_dir in self.spawn_dirs]
            self.output_images = [spawn_dir + '/sphere/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]
            
        else: raise ValueError("Trying to call dataset class for other purposes is not allowed")
        
        self.input_spawns = np.array(self.input_images)
        self.output_spawns = np.array(self.output_images)
        
        # # Selected randomly to train just on a single spawn point
        # self.input_spawns = np.array(self.input_images)[:10]
        # self.output_spawns = np.array(self.output_images)[:10]
        
        # configuring relevant resolution for inward and outward facing cameras
        
        self.context_resolution = (self.view_sampler.cfg.input_context_resolution, self.view_sampler.cfg.input_context_resolution)
        self.target_resolution = (self.view_sampler.cfg.output_target_resolution, self.view_sampler.cfg.output_target_resolution)
        
        #######################################################################
        # Below is the important flag changing workflow of several blocks
        self.augment_flag = self.stage == 'train' and self.cfg.train_view_sampler.augment
        
        # Here we are loading all the data into RAM 
        _ = [self.load_example_id(idx) for idx in range(0, len(self.input_spawns))]
        print(f"Carla Dataset, initialized for {self.stage} stage, will use # {len(self.input_spawns)} spawns with augmentation = {self.augment_flag}")
        print(f"Training towns: {training_towns}, Testing towns: {testing_towns}")
    
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

    def get_example_id(self, index):
        intrin_path = self.input_spawns[index]
        example_id = intrin_path[:find_nth_reverse(intrin_path, '/', 3)]
        return example_id
    
    def load_example_id(self, index):
        
        example_id = self.get_example_id(index)
        
        input_transforms = self.input_spawns[index]
        output_transforms = self.output_spawns[index]
        
        if not hasattr(self, "all_texture_context"):
            
            self.all_texture_context = {}
            self.all_texture_target = {}
            
            self.intrinsics_context = {}
            self.intrinsics_target = {}
            
            self.extrinsics_context = {}
            self.extrinsics_target = {}
            
        if example_id not in self.all_texture_context.keys():
            
            self.all_texture_context[example_id] = []
            self.all_texture_target[example_id] = []
            
            self.intrinsics_context[example_id] = []
            self.intrinsics_target[example_id] = []
            
            self.extrinsics_context[example_id] = []
            self.extrinsics_target[example_id] = []
            #
            # # obtaining intrinsics & extrinsics for context & target & render cameras
            context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices = readPixelSplatCamera(input_transforms, resolution=self.view_sampler.cfg.input_context_resolution, 
                                                                                                                 near=self.cfg.z_near, far=self.cfg.z_far)
            target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = readPixelSplatCamera(output_transforms, resolution=self.view_sampler.cfg.output_target_resolution, 
                                                                                                              near=self.cfg.z_near, far=self.cfg.z_far)
            
            # Adding 6 Ego Vehicle camera views 
            for image_path, intrins, extrins in zip(context_image_paths, context_intrinsics_matrices, context_extrinsics_matrices):
                self.all_texture_context[example_id].append(image_path)
                self.intrinsics_context[example_id].append(intrins)
                self.extrinsics_context[example_id].append(extrins)
            # stacking intrinsics and extrinsics individually for convenience
            self.intrinsics_context[example_id] = torch.stack(self.intrinsics_context[example_id]).cpu()
            self.extrinsics_context[example_id] = torch.stack(self.extrinsics_context[example_id]).cpu()
            
            # Adding all target camera views 
            for image_path, intrins, extrins in zip(target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices):
                self.all_texture_target[example_id].append(image_path)
                self.intrinsics_target[example_id].append(intrins)
                self.extrinsics_target[example_id].append(extrins)
            # stacking intrinsics and extrinsics individually for convenience
            self.intrinsics_target[example_id] = torch.stack(self.intrinsics_target[example_id]).cpu()
            self.extrinsics_target[example_id] = torch.stack(self.extrinsics_target[example_id]).cpu()
    
    def __getitem__(self, index):
        example_id = self.get_example_id(index)
        # #
        # # view sampler needs to know context and target view extrinsics for sampling strategy
        index_context, index_target = self.view_sampler.sample("SEED", self.extrinsics_context[example_id], 
                                                               self.extrinsics_target[example_id])
        
        #######################################################################
        #
        #######################################################################
        ############### Loading Context/Conditional Information ###############
        context_images = [img_path_to_Torch(image_path, self.context_resolution)
                          for image_path in np.array(self.all_texture_context[example_id])[index_context.numpy()]]
        context_images = torch.stack(context_images).float()
        context_extrinsics = self.extrinsics_context[example_id][index_context.numpy()]
        context_intrinsics = self.intrinsics_context[example_id][index_context.numpy()]
        #######################################################################
        ################ Loading Inference Target Information #################
        # Reading images
        target_images = [img_path_to_Torch(image_path, self.target_resolution)
                         for image_path in np.array(self.all_texture_target[example_id])[index_target.numpy()]]
        target_images = torch.stack(target_images).float()
        # Reading depth maps
        target_depths = [depth_path_to_Torch(image_path[:image_path.rfind('_')] + "_depth.png", self.target_resolution)
                         for image_path in np.array(self.all_texture_target[example_id])[index_target.numpy()]]
        target_depths = torch.stack(target_depths).float()
        # Reading Camera params
        target_extrinsics = self.extrinsics_target[example_id][index_target.numpy()]
        target_intrinsics = self.intrinsics_target[example_id][index_target.numpy()]
        
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
