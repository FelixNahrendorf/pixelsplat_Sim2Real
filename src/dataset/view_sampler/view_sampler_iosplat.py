from dataclasses import dataclass
from typing import Literal

import random
import torch
import numpy as np
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler
from itertools import islice, cycle

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@dataclass
class ViewSamplerIOsplatCfg:
    name: Literal["train_carla_view", "eval_carla_view"]
    num_context_views: int
    num_target_views: int
    augment: bool
    augment_p: float
    augment_mask_count: int
    context_views: list[int] | None
    target_views: list[int] | None
    #  #
    input_context_resolution: int
    output_target_resolution: int


class ViewSamplerIOsplat(ViewSampler[ViewSamplerIOsplatCfg]):
    def sample(
        self,
        scene,
        extrinsics_context: Float[Tensor, "cview 4 4"],
        extrinsics_target: Float[Tensor, "tview 4 4"],
        experiment: str = "ego-exo",
        device: torch.device = torch.device("cpu"),
        use_nuscene_context: bool = False,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Sample target views depending on the training
            versus testing modes (target views are kept constant during inference)
            Context views are always set to conditional 6 images on Ego Vehicle
        """
        temperature = 1 

        ### Explanation on temperature parameter in softmax function:
        #temperature = 1: Normal softmax behavior, 
        #temperature > 1 (e.g., 2, 5): Makes the distribution more uniform/smooth
        #temperature < 1 (e.g., 0.5, 0.1): Makes the distribution sharper/more peaked
        #temperature → 0: Nearly deterministic selection of only the most similar views
        #temperature → ∞: Uniform random sampling (ignores similarity entirely)

        if self.cfg.num_context_views<6:
            Choice = [0, 1, 2, 3, 4, 5] #Choice = [0, 1, 5, 3, 4, 2]
            start = random.randint(0, len(Choice) - 1)
            index_context = torch.tensor(list(islice(cycle(Choice), start, start + self.cfg.num_context_views))).to(dtype=torch.int64)
        else:
            #index_context = torch.from_numpy(np.array([0, 1, 5, 3, 4, 2])).to(dtype=torch.int64, device=device)
            
            index_context = torch.from_numpy(np.array([0, 1, 2, 3, 4, 5])).to(dtype=torch.int64, device=device)
            
        # #
        # # We will sample only those target views that are 'similar' to the context views
        # # where the similarity is determined by the direction of the cameras
        # We will sample only from 20 views for test&val stages
        context_camera_directions = extrinsics_context[index_context, :, 2]
        target_camera_directions = extrinsics_target[:, :, 2]
        context_target_similarity = np.einsum("ij,lj-> il", context_camera_directions, target_camera_directions)
        # we are multiplying similarity scores by temperature to increase sampling of target views similar to context views more 
        target_sample_weight_map = np.array([softmax(temperature*row) for row in context_target_similarity])
        target_sample_weight = np.max(target_sample_weight_map, axis=0)
        target_sample_weight = softmax(temperature*target_sample_weight)

        ###ego-ego/ego-exo mixed training

        if experiment == "ego-exo-mixed":

            if self.stage=='test' or self.stage=='val':
                # If the (hardcoded) target views are not None, then use them  
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                # If not, then randomly select them
                else:
                    assert self.cfg.num_target_views<=20 
                    index_target = torch.from_numpy(np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, 
                                                                    replace=False)).to(dtype=torch.int64) 
            elif self.stage == 'train':
                index_target = torch.from_numpy(np.random.choice(np.arange(0, 98), size=self.cfg.num_target_views, 
                                                                replace=False)).to(dtype=torch.int64) 

            else: raise KeyError("Called dataset with wrong stage argument ... ")
        
        elif experiment == "ego-exo-mixed-domain":
            print(f"[DEBUG VIEW SAMPLER] use_nuscene_context: {use_nuscene_context}, extrinsics_context shape: {extrinsics_context.shape}")
            
            if use_nuscene_context and extrinsics_context.shape[0] > 6: 
                ### Use 5 SEED4D + 1 nuScenes every 10th scene###
                #nuscene_indices = list(range(6))
                #replace_position = random.randint(0, 5)
                #nuscene_start_idx = 6
                #nuscene_idx = nuscene_start_idx + replace_position
                #nuscene_indices[replace_position] = nuscene_idx
                #print(f"[NUSCENES MIX] Using mixed context: SEED4D indices {[i for i in nuscene_indices if i < 6]} + nuScenes index {nuscene_idx} at position {replace_position}")

                ### Use 6 nuScenes every 10th scene for context and target###
                nuscene_indices = [6,7,8,9,10,11]
                print(f"[NUSCENES ONLY] Using NUSCENES-only context", "index_context", index_context)
                index_context = torch.tensor(nuscene_indices, dtype=torch.int64, device=device)

                nuscene_target_indices = [98,99,100,101,102,103]
                index_target = torch.tensor(nuscene_target_indices, dtype=torch.int64, device=device) 
                print(f"[NUSCENES ONLY] Using NUSCENES-only target", "index_target", index_target )

                if self.stage=='test' or self.stage=='val':
                    # If the (hardcoded) target views are not None, then use them  
                    if self.cfg.target_views is not None:
                        assert len(self.cfg.target_views) == self.cfg.num_target_views
                        index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                    else:
                        assert self.cfg.num_target_views <= 20 
                        nuscene_target_indices = [99, 100, 101, 102, 103, 104]
                        #index_target = torch.from_numpy(
                        #    np.array(nuscene_target_indices[:self.cfg.num_target_views])).to(dtype=torch.int64, device=device)
                        index_target = torch.tensor(nuscene_target_indices, dtype=torch.int64, device=device) 
                        
                elif self.stage == 'train':  
                    nuscene_target_indices = [99, 100, 101, 102, 103, 104]
                    # This is correct - already uses num_target_views
                    index_target = torch.from_numpy(
                        np.random.choice(nuscene_target_indices, size=self.cfg.num_target_views, replace=False)).to(dtype=torch.int64, device=device)
                
            else:
                # Use only SEED4D
                if self.cfg.num_context_views <= 6:
                    index_context = torch.arange(0, self.cfg.num_context_views, 
                                                dtype=torch.int64, device=device)
                else:
                    index_context = torch.arange(0, 6, dtype=torch.int64, device=device)
                print(f"[SEED4D ONLY] Using SEED4D-only context")
                
                # Sample target indices only for SEED4D (not nuScenes)
                if self.stage=='test' or self.stage=='val':
                    # If the (hardcoded) target views are not None, then use them  
                    if self.cfg.target_views is not None:
                        assert len(self.cfg.target_views) == self.cfg.num_target_views
                        index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                    # If not, then randomly select them
                    else:
                        assert self.cfg.num_target_views<=20 
                        index_target = torch.from_numpy(np.random.choice(np.arange(0, 98), size=self.cfg.num_target_views, 
                                                                        replace=False)).to(dtype=torch.int64) 
                elif self.stage == 'train': #80 sphere target views + 3*6 SEED4D ego views = 98 
                    index_target = torch.from_numpy(np.random.choice(np.arange(0, 98), size=self.cfg.num_target_views, 
                                                                    replace=False)).to(dtype=torch.int64)
                # ======================================================================
                    
        elif experiment == "ego-exo":
            if self.stage=='test' or self.stage=='val':
                # If the (hardcoded) target views are not None, then use them  
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                else:
                    assert self.cfg.num_target_views<=20 
                    index_target = torch.from_numpy(np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, 
                                                                    replace=False)).to(dtype=torch.int64)
            elif self.stage == 'train':
                index_target = torch.from_numpy(np.random.choice(np.arange(0, 80), size=self.cfg.num_target_views, 
                                                                replace=False, p=target_sample_weight)).to(dtype=torch.int64) 
        elif experiment == "ego-ego":
            # If the (hardcoded) target views are not None, then use them  
            if self.cfg.target_views is not None:
                assert len(self.cfg.target_views) == self.cfg.num_target_views
                index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
            else:
                ### no randomization for ego-ego testing for better visual comaprison of the images
                #index_target = torch.tensor([0,1,2,3,4,5])
                index_target = torch.tensor([0,1,2,3,5,6])
        else: raise KeyError("Called dataset with wrong experiment argument ... ")

        return index_context, index_target
    
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views
    
    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views