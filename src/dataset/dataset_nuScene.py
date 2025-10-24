'''
This Dataloader combines the Nuscene and SEED4D Dataset to perform mixed domain training and testing of pixelsplat
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
from typing import Literal, NamedTuple, List
from queue import Queue
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

from .nuscene_reader import desired_sensor_names, CameraInfo, frame_seq_transform, CAM2RADARS, STATIONARY_CATEGORIES
from ..misc.general_utils import img_path_to_Torch, depth_path_to_Torch
from .dataset_readers import readPixelSplatCamera

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords, generate_record
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion.quaternion import Quaternion

NUSCENE_DATA_DIR = "/app/datasets/nuscenes_full/" 
assert NUSCENE_DATA_DIR is not None, "Update the location of the NUSCENE Dataset"

SEED4D_DATASET_ROOT = '/app/inputs/seed4d/data/data_diverse/static/'  #'/app/data/seed4d/static/'   ## initial dataset
assert SEED4D_DATASET_ROOT is not None, "Update the location of the SEED4D Dataset"

#SEED4D_DATASET_ROOT_10_SCENES = '/app/code/seed4d/data/data_sensititvity_analysis_baseline_nuscene_adjusted/static/'  ## initial dataset
#assert SEED4D_DATASET_ROOT_10_SCENES is not None, "Update the location of the SEED4D Dataset"
    
@dataclass
class Dataset_NUSCENECfg(DatasetCfgCommon):
    name: Literal["nuscene"]
    train_view_sampler: ViewSamplerCfg
    eval_view_sampler: ViewSamplerCfg
    nuscene_lidar_point_num: int
    max_fov: float
    z_near: float
    z_far: float
    experiment: str
    training_towns: List[str] = None  # Add training towns configuration
    testing_towns: List[str] = None   # Add testing towns configuration
    
class Dataset_NUSCENE(Dataset):
    cfg: Dataset_NUSCENECfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: Dataset_NUSCENECfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()
        self.view_sampler = view_sampler        
        self.td = self.view_sampler.cfg.nuscene_td
        # # # frame_seq_size acts as the number of frames
        # # # considered during modelling a single frame 
        self.frame_seq_size = int(2 * self.td + 1)
        self.version = self.view_sampler.cfg.nuscene_version
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #######################################################
        # initialize an instance of nuScenes data
        self.nusc = NuScenes(version=self.version, dataroot=NUSCENE_DATA_DIR)
        self.all_scenes = [record for record in self.nusc.scene]
        self.sample_scenes = []
        # # # frame information will be dictionary storing camera 
        # # # parameters and image path making it faster for the  
        # # # dataloader to retrieve a sequence of frames
        self.all_frame_information = dict()
        #######################################################
        # prepare a symbol table to map a split to its scenes
        all_splits = create_splits_scenes()
        if self.version == 'v1.0-mini':
            self.usable_splits = {
                    'train': all_splits['mini_train'],
                    'val': all_splits['mini_val']}
        elif self.version == 'v1.0-trainval':
            self.usable_splits = {
                    'train': all_splits['train'],
                    'val': all_splits['val']}
        elif self.version == 'v1.0-test':
            self.usable_splits = {'test': all_splits['test']}
        else: raise NotImplementedError
        #######################################################
        # print("\n\n", self.stage, self.usable_splits, "\n\n")
        
        data_dir_naming = '/ClearNoon/vehicle.audi.tt/'
        
        # Use configuration values or default fallback
        #training_towns = ['02'] #self.cfg.training_towns if self.cfg.training_towns is not None else ['02']
        testing_towns = self.cfg.testing_towns if self.cfg.testing_towns is not None else ['02']
        
        if (self.stage == 'train'):
            self.scene_names = self.usable_splits["train"]
            ### copied from seed4d dataset
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns] #data/seed4d/static/Town02/ClearNoon
            self.spawn_dirs =  [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle')  for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
            self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_train.json' for spawn_dir in self.spawn_dirs]
        elif (self.stage == 'val'): # val stands for validation
            self.scene_names = self.usable_splits["val"]
            ### copied from seed4d dataset
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in training_towns]
            self.spawn_dirs =  [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle')  for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
            self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]
        elif (self.stage == 'test'): 
            self.scene_names = self.usable_splits["test"]
            ### copied from seed4d dataset
            self.parent_dirs = [SEED4D_DATASET_ROOT + 'Town' + town + data_dir_naming for town in testing_towns]
            self.spawn_dirs =  [str_list_concat(spawns_dir, spawns_dir, 'step_0/ego_vehicle')  for spawns_dir in self.parent_dirs]
            self.spawn_dirs = list(itertools.chain.from_iterable(self.spawn_dirs))
            random.shuffle(self.spawn_dirs) 
            self.output_images = [spawn_dir + '/sphere_invisible/transforms/transforms_ego_test.json' for spawn_dir in self.spawn_dirs]
        
        else: raise ValueError("Trying to call dataset class for other purposes is not allowed")
        #######################################################
        # # # Here we obtain a list of all scenes for stage (train/val/test)
        # # # self.sample_scenes is already storing scene tokens
        for sample_scene in self.all_scenes:
            if sample_scene["name"] in self.scene_names:
                self.sample_scenes.append(sample_scene)
        # # # but we need to also consider frames per scene s.t. 
        # # # we are in the same setup as SEED4D to finetune sshELF
        #######################################################
        self.all_frame_sequences = []
        self.all_frame_tokens = []
        for sample_scene in self.sample_scenes:
            per_scene_queue = Queue(maxsize = self.frame_seq_size)
            first_sample_token = sample_scene["first_sample_token"]
            ####################################################################
            # # # obtaining location of the ego vehicle at the first frame
            first_frame_information = self.nusc.get("sample", first_sample_token)
            first_frame_information = first_frame_information["data"]
            first_frame_front_cam = self.nusc.get("sample_data", 
                                                first_frame_information['CAM_FRONT'])
            first_frame_ego_pose = self.nusc.get(table_name="ego_pose", 
                                                token=first_frame_front_cam["ego_pose_token"])
            first_frame_translation = np.array(first_frame_ego_pose["translation"])
            ####################################################################
            # last_sample_token = sample_scene["last_sample_token"]
            current_sample_token = first_sample_token
            while True:
                # # # we will store first frame translation per sequence in 
                # # # order to 'normalize' w.r.t translation vector
                self.all_frame_tokens.append(current_sample_token)
                #############################################################################################
                nuscene_frame_information = self.nusc.get("sample", current_sample_token)
                per_scene_queue.put(current_sample_token)            # # this placement is better suited
                if per_scene_queue.qsize() == self.frame_seq_size:
                    # # # brute force way of copying queue without changing 
                    # # # original queue at hand
                    sample_queue = Queue(maxsize = self.frame_seq_size)
                    for _ in range(self.frame_seq_size):
                        frame_token = per_scene_queue.get()
                        sample_queue.put(frame_token)
                        per_scene_queue.put(frame_token)
                    self.all_frame_sequences.append(sample_queue)
                    _ = per_scene_queue.get()
                # # # ending the loop when next frame is empty
                current_sample_token = nuscene_frame_information["next"]
                if current_sample_token == "": break
        # # # thus each element in sample_frames will store
        # # # self.frame_seq_size frame tokens --> middle token
        # # # will act as a reference token s.t. we will sample
        # # # reference frames from that middle frame token

        self.output_spawns = np.array(self.output_images)
        #######################################################
        self.context_resolution = (self.view_sampler.cfg.input_context_resolution, self.view_sampler.cfg.input_context_resolution)
        self.target_resolution = (self.view_sampler.cfg.output_target_resolution, self.view_sampler.cfg.output_target_resolution)
        #######################################################################
        # Below is the important flag changing workflow of several blocks
        self.augment_flag = self.stage == 'train' and self.cfg.train_view_sampler.augment
        #######################################################################
        # Clear any existing cached data to ensure camera order changes take effect
        self.all_frame_information.clear() ##DEBUG
        # Here we are loading all the data into RAM 
        _ = [self.load_example(frame_token) for frame_token in self.all_frame_tokens]
        _ = [self.load_example_id(idx) for idx in range(0, len(self.output_spawns))]
        
        print(f"nuScene Dataset, initialized for {self.stage} stage, will use # {len(self.all_frame_sequences)} spawns with augmentation = {self.augment_flag}")
    
    def __len__(self):
        return len(self.all_frame_sequences)
    
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

    ### copied from seed4d
    def get_example_id(self, index):
        intrin_path = self.output_spawns[index]
        example_id = intrin_path[:find_nth_reverse(intrin_path, '/', 3)]
        return example_id

    ### copied from seed4d
    def load_example_id(self, index):
        
        example_id = self.get_example_id(index)
        
        output_transforms = self.output_spawns[index]
        
        if not hasattr(self, "all_texture_target"):
            
            self.all_texture_target = {}
            self.intrinsics_target = {}
            self.extrinsics_target = {}
            
        if example_id not in self.all_texture_target.keys():
            
            self.all_texture_target[example_id] = []
            self.intrinsics_target[example_id] = []
            self.extrinsics_target[example_id] = []

            
            #print('self.view_sampler.cfg.output_target_resolution', self.view_sampler.cfg.output_target_resolution) ###DEBUG #256

            # # obtaining intrinsics & extrinsics for context & target & render cameras ###maybe resolution=self.view_sampler.cfg.output_target_resolution need to be modified
            target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices = readPixelSplatCamera(output_transforms, resolution=self.view_sampler.cfg.output_target_resolution, 
                                                                                                              near=self.cfg.z_near, far=self.cfg.z_far)
            
            
            # Adding ALL target camera views (no filtering applied)
            for image_path, intrins, extrins in zip(target_image_paths, target_intrinsics_matrices, target_extrinsics_matrices):
                self.all_texture_target[example_id].append(image_path)
                self.intrinsics_target[example_id].append(intrins)
                self.extrinsics_target[example_id].append(extrins)
            # stacking intrinsics and extrinsics individually for convenience
            self.intrinsics_target[example_id] = torch.stack(self.intrinsics_target[example_id]).cpu()
            self.extrinsics_target[example_id] = torch.stack(self.extrinsics_target[example_id]).cpu()
    
    '''def adjust_intrinsics(self, features, intrinsics, final_res): ## could possibly be removed
        # # #
        _, _, h, w = features.shape
        assert h == w # # # we used to work with square images
        assert final_res[0] == final_res[1]
        resize_ratio_w = w / final_res[0] # / w
        resize_ratio_h = h / final_res[1] # / h
        # # # only focal lengths will be affected
        adjusted_intrinsics = intrinsics.clone()
        adjusted_intrinsics[:, 0, 0] *= resize_ratio_w
        adjusted_intrinsics[:, 1, 1] *= resize_ratio_h
        return F.interpolate(features, final_res, mode="bilinear", align_corners=True), adjusted_intrinsics'''
        
    def load_example(self, frame_token):
        frame_information = []      # # # better to keep 'frame_information' in a list format
        # # # while the reading order of cameras can be found in nuscene_reader.py
        nuscens_frame_information = self.nusc.get("sample", frame_token)
        sample_frame_info = nuscens_frame_information["data"]
        frame_LIDAR_token = sample_frame_info["LIDAR_TOP"]
        for sensor in desired_sensor_names:
            sensor_data = self.nusc.get("sample_data", sample_frame_info[sensor])
#until here order is correct
            #print('sample_frame_info[sensor]', sample_frame_info[sensor]) ###DEBUG
            # # retaining ego vehicle location w.r.t global coordinate system
            ego_pose_information = self.nusc.get(table_name="ego_pose", token=sensor_data["ego_pose_token"])
            ego_pose_rotation = Quaternion(ego_pose_information["rotation"])
            ego_pose_translation = np.array(ego_pose_information["translation"])

            # # # # # # # # # # # # # # # # # #
            ego_transform_matrix = transform_matrix(ego_pose_translation, ego_pose_rotation)

            # # # # # # # # # # # # # # # # # #
            # # retrieving sensor placement w.r.t ego vehicle
            sensor_pose_information = self.nusc.get(table_name="calibrated_sensor", 
                                                    token=sensor_data["calibrated_sensor_token"])
            sensor_intrinsic_matrix = np.array(sensor_pose_information["camera_intrinsic"])
            sensor_pose_rotation = Quaternion(sensor_pose_information["rotation"])
            sensor_pose_translation = np.array(sensor_pose_information["translation"])
            # # # # # # # # # # # # # # # # # #
            sensor_transform_matrix = transform_matrix(sensor_pose_translation, sensor_pose_rotation)       
            #sensor_transform_matrix = ego_transform_matrix @ sensor_transform_matrix

            # # # # # # # # # # # # # # # # # #       
            sensor_file_path = NUSCENE_DATA_DIR + sensor_data["filename"]
            sensor_information = CameraInfo(intrinsic=sensor_intrinsic_matrix, extrinsic=sensor_transform_matrix, 
                                            image_path=sensor_file_path, width=sensor_data["width"], 
                                            height=sensor_data["height"], name=sensor_data["channel"], 
                                            ego_T=ego_pose_translation)
            frame_information.append(sensor_information)
            #print('sensor_information', sensor_information) ###DEBUG
#until here order is correct            
        ####################################################################
        self.all_frame_information[frame_token] = {"frame_info": frame_information, 
                                                   "lidar_token": frame_LIDAR_token}
    ########################################################################
    def get_frame_seq(self, frame_seq):
        # # # sanity check
        # print(f"\n\nError happens below between {frame_seq.qsize()} and {self.frame_seq_size} shapes")
        assert frame_seq.qsize() == self.frame_seq_size                                                                                                                                                                                                                                                                          
        # # #
        sequence_frame_info = []
        sequence_token_que = Queue(self.frame_seq_size)                                                                                                                                                                                                                                                                      
        for _ in range(self.frame_seq_size):                                                                                                                                                                                                                                                                       
                frame_token = frame_seq.get()                                                                                                                                                                                                                                                                        
                sequence_token_que.put(frame_token)                                                                                                                                                                                                                                                                      
                sequence_frame_info.append(self.all_frame_information[frame_token])   
        points, features, extrinsics, intrinsics = \
            frame_seq_transform(sequence_frame_info, nuscene_loader=self.nusc, resolution=self.context_resolution[0], 
                                near=self.cfg.z_near, far=self.cfg.z_far, nuscene_lidar_point_num=self.cfg.nuscene_lidar_point_num)                                                                                                                                                                                                                
        return sequence_token_que, points, features, extrinsics, intrinsics
    
    
    




    def __getitem__(self, index):
        example_id = self.get_example_id(index)
        ###maybe necessary to add in a modified way:
        '''index_context, index_target = self.view_sampler.sample("SEED", self.extrinsics_context[example_id], 
                                                               self.extrinsics_target[example_id])'''
        ###
        sample_sequence = self.all_frame_sequences[index]
        sample_sequence, points, features, extrinsics, intrinsics = self.get_frame_seq(sample_sequence)

        # Get the actual number of SEED4D target cameras available
        num_seed4d_targets = len(self.all_texture_target[example_id])

        index_context, index_target, reference_frame, target_frame = self.view_sampler.sample(extrinsics, stage=self.stage, available_target_views=num_seed4d_targets)
        seq, view, c, h, w = features.shape
        # # # sample_sequence should be kept in dataset
        self.all_frame_sequences[index] = sample_sequence
        #######################################################################
        ############### Loading Context/Conditional Information ###############
        context_images = features[reference_frame][index_context]
        context_extrinsics = extrinsics[reference_frame][index_context]
        context_intrinsics = intrinsics[reference_frame][index_context]
        #print('context_images shape:', context_images.shape)         #context_images shape: context_images shape: torch.Size([6, 3, 256, 256])
        #print('context_intrinsics shape:', context_intrinsics.shape) #context_intrinsics shape: context_intrinsics shape: torch.Size([6, 3, 3])
        #print('context_extrinsics shape:', context_extrinsics.shape) #context_extrinsics shape: context_extrinsics shape: torch.Size([6, 4, 4])
        
        # Apply reordering using advanced indexing
        #camera_reorder_mapping = [0, 1, 5, 3, 4, 2]
        #context_images = context_images[camera_reorder_mapping]
        #context_extrinsics = context_extrinsics[camera_reorder_mapping]
        #context_intrinsics = context_intrinsics[camera_reorder_mapping]
        #index_context = index_context[camera_reorder_mapping]


        #######################################################################
        ################ Loading Inference Target Information #################
        '''target_images = features[target_frame][index_target]
        target_extrinsics = extrinsics[target_frame][index_target]
        target_intrinsics = intrinsics[target_frame][index_target]
        target_images, target_intrinsics = self.adjust_intrinsics(target_images, target_intrinsics, self.target_resolution)'''
        
        #print('target_images shape:', target_images.shape)         #target_images shape: torch.Size([6, 3, 256, 256])
        #print('target_intrinsics shape:', target_intrinsics.shape) #target_intrinsics shape: torch.Size([6, 3, 3])
        #print('target_extrinsics shape:', target_extrinsics.shape) #target_extrinsics shape: torch.Size([6, 4, 4])

        ### copied from seed4d

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
                    "scene": "nuScene",
                    "point_cloud": points}
     
        print("=== FINAL DATA FED TO MODEL ===")
        print('context_resolution:', self.context_resolution) ###DEBUG
        print('context_resolution[0]:', self.context_resolution[0]) ###DEBUG
        print(f"Resolution: {self.view_sampler.cfg.output_target_resolution}")
        #print(f"Output transforms file: {output_transforms}")
        print(f"Index target: {index_target.numpy()}")
        print(f"Target intrinsics shape: {self.intrinsics_target[example_id].shape}")

        print('index_context.numpy():', index_context.numpy())
        print('index_context:', index_context)
        print('context_images', context_images)

        print("Context intrinsics shape:", example['context']['intrinsics'].shape)
        print("Context intrinsics:\n", example['context']['intrinsics'])
        print("Context extrinsics shape:", example['context']['extrinsics'].shape)
        print("Context extrinsics:\n", example['context']['extrinsics'])
        print("Target intrinsics shape:", example['target']['intrinsics'].shape) 
        print("Target intrinsics:\n", example['target']['intrinsics'])
        print("Target extrinsics shape:", example['target']['extrinsics'].shape)
        print("Target extrinsics:\n", example['target']['extrinsics'])
        print("="*50)
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

def path_to_torch(file_path):
        # print("Will process path: ", file_path)
        sample_path_splits = file_path.split('/')
        
        npy_path = PROCESSED_DATASET + sample_path_splits[3] + '/' + sample_path_splits[6] + '/' \
                                + sample_path_splits[7] + '/' + sample_path_splits[8] + '/' \
                                + sample_path_splits[10][:sample_path_splits[10].rfind('.')] + '.npy'
        # print(torch.from_numpy(np.load(npy_path)).shape)
        return torch.from_numpy(np.load(npy_path))
    
def batch_collate_func(batch):
    batch_example = {
                    "context": {
                        "extrinsics": torch.stack([batch_elem["context"]["extrinsics"] for batch_elem in batch]),
                        "intrinsics": torch.stack([batch_elem["context"]["intrinsics"] for batch_elem in batch]),
                        "image": torch.stack([batch_elem["context"]["image"] for batch_elem in batch]),
                        "near": torch.stack([batch_elem["context"]["near"] for batch_elem in batch]),
                        "far": torch.stack([batch_elem["context"]["far"] for batch_elem in batch]),
                        "index": torch.stack([batch_elem["context"]["index"] for batch_elem in batch])},
                    "target": {
                        "extrinsics": torch.stack([batch_elem["target"]["extrinsics"] for batch_elem in batch]),
                        "intrinsics": torch.stack([batch_elem["target"]["intrinsics"] for batch_elem in batch]),
                        "image": torch.stack([batch_elem["target"]["image"] for batch_elem in batch]),
                        "depth": torch.stack([batch_elem["target"]["depth"] for batch_elem in batch]),
                        "near": torch.stack([batch_elem["target"]["near"] for batch_elem in batch]),
                        "far": torch.stack([batch_elem["target"]["far"] for batch_elem in batch]),
                        "index": torch.stack([batch_elem["target"]["index"] for batch_elem in batch])},
                    "render": {
                        "extrinsics": torch.stack([batch_elem["render"]["extrinsics"] for batch_elem in batch]),
                        "intrinsics": torch.stack([batch_elem["render"]["intrinsics"] for batch_elem in batch]),
                        "image": torch.stack([batch_elem["render"]["image"] for batch_elem in batch]),
                        "depth": torch.stack([batch_elem["render"]["depth"] for batch_elem in batch]),
                        "near": torch.stack([batch_elem["render"]["near"] for batch_elem in batch]),
                        "far": torch.stack([batch_elem["render"]["far"] for batch_elem in batch]),
                        "index": torch.stack([batch_elem["render"]["index"] for batch_elem in batch])}}
    #print ('batch_example', batch_example) ###DEBUG
    return batch_example

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

### new implementation of transformation ###

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

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    new transformation to match the format pixelsplat was trained on (format of seed4d/carla generator)
    """

    # Apply x-axis flip
    x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
    original_quaternion = rotation * x_axis_flip
        
    # Apply coordinate transformation
    transformed_translation = apply_coordinate_transformation(translation)
    transformed_quaternion = apply_rotation_transformation(original_quaternion)
        
    # Create transformation matrix
    transformed_transform_matrix = quaternion_to_transform_matrix(
        transformed_quaternion, transformed_translation
    )

    transformed_transform_matrix[0,1] *= -1  
    transformed_transform_matrix[0,2] *= -1
    transformed_transform_matrix[1,1] *= -1
    transformed_transform_matrix[1,2] *= -1
    transformed_transform_matrix[2,1] *= -1
    transformed_transform_matrix[2,2] *= -1

    return transformed_transform_matrix