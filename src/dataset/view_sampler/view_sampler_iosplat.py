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
            Choice = [0, 1, 2, 3, 4, 5] 
            start = random.randint(0, len(Choice) - 1)
            index_context = torch.tensor(list(islice(cycle(Choice), start, start + self.cfg.num_context_views))).to(dtype=torch.int64)
        else:
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
                                                                    replace=False)).to(dtype=torch.int64) #this may be needed: .to(dtype=torch.int64, device=device) 
            elif self.stage == 'train':
                index_target = torch.from_numpy(np.random.choice(np.arange(0, 98), size=self.cfg.num_target_views, 
                                                                replace=False)).to(dtype=torch.int64) #this may be needed: .to(dtype=torch.int64, device=device) 

            else: raise KeyError("Called dataset with wrong stage argument ... ")
        
        elif experiment == "ego-exo-mixed-domain":

            nuscene_context_indices = [6,7,8,9,10,11]
            nuscene_target_indices = [98,99,100,101,102,103]

            if use_nuscene_context:
                ### Use 6 nuScene views every 10th scene for context and target###
                index_context = torch.tensor(nuscene_context_indices, dtype=torch.int64, device=device)

                if self.stage=='test' or self.stage=='val':
                    # If the (hardcoded) target views are not None, then use them  
                    if self.cfg.target_views is not None:
                        assert len(self.cfg.target_views) == self.cfg.num_target_views
                        index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                    # If not, then randomly select them
                    else:
                        #assert self.cfg.num_target_views<=20 
                        #index_target = torch.from_numpy(np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, 
                        #                                                replace=False)).to(dtype=torch.int64) #this may be needed: .to(dtype=torch.int64, device=device)
                        nuscene_target_indices_extended = (nuscene_target_indices * 3)[:18] + random.choices(nuscene_target_indices, k=2)
                        index_target = torch.tensor(nuscene_target_indices_extended, dtype=torch.int64, device=device)

                elif self.stage == 'train':
                    index_target = torch.from_numpy(np.random.choice(nuscene_target_indices, size=self.cfg.num_target_views, 
                                                                    replace=False)).to(dtype=torch.int64, device=device)
  
            else:
                ### Use SEED4D views only
                if self.cfg.num_context_views <= 6:
                    index_context = torch.arange(0, self.cfg.num_context_views, 
                                                dtype=torch.int64, device=device)
                else:
                    index_context = torch.arange(0, 6, dtype=torch.int64, device=device)
                print(f"[SEED4D ONLY] Using SEED4D-only context")
                
                # Sample target indices only for SEED4D 
                if self.stage=='test' or self.stage=='val':
                    # If the (hardcoded) target views are not None, then use them  
                    if self.cfg.target_views is not None:
                        assert len(self.cfg.target_views) == self.cfg.num_target_views
                        index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                    # If not, then randomly select them
                    else:
                        assert self.cfg.num_target_views<=20 
                        index_target = torch.from_numpy(np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, 
                                                                        replace=False)).to(dtype=torch.int64, device=device) 
                elif self.stage == 'train': #80 sphere target views + 3*6 SEED4D ego views = 98 
                    index_target = torch.from_numpy(np.random.choice(np.arange(0, 98), size=self.cfg.num_target_views, 
                                                                    replace=False)).to(dtype=torch.int64, device=device)
                # ======================================================================
        elif experiment == "ego-ego-nuscenes":

            nuscene_context_indices = [6,7,8,9,10,11]
            nuscene_target_indices = [98,99,100,101,102,103]

            if use_nuscene_context:
                ### Use 6 nuScene views every scene for context and target###
                index_context = torch.tensor(nuscene_context_indices, dtype=torch.int64, device=device)

                if self.stage=='test' or self.stage=='val':
                    # If the (hardcoded) target views are not None, then use them  
                    if self.cfg.target_views is not None:
                        assert len(self.cfg.target_views) == self.cfg.num_target_views
                        index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                    # If not, then randomly select them
                    else:
                        #assert self.cfg.num_target_views<=20 
                        #index_target = torch.from_numpy(np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, 
                        #                                                replace=False)).to(dtype=torch.int64) #this may be needed: .to(dtype=torch.int64, device=device)
                        #nuscene_target_indices_extended = (nuscene_target_indices * 3)[:18] + random.choices(nuscene_target_indices, k=2)
                        index_target = torch.tensor(nuscene_target_indices, dtype=torch.int64, device=device)

                elif self.stage == 'train':
                    index_target = torch.from_numpy(np.random.choice(nuscene_target_indices, size=self.cfg.num_target_views, 
                                                                    replace=False)).to(dtype=torch.int64, device=device)
        elif experiment == "ego-exo-nuscenes":

            nuscene_context_indices = [6,7,8,9,10,11]
                # Sample target indices only for SEED4D 
            if self.stage=='test' or self.stage=='val':
                # If the (hardcoded) target views are not None, then use them  
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                # If not, then randomly select them
                else:
                    assert self.cfg.num_target_views<=20 
                    index_target = torch.from_numpy(np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, 
                                                                    replace=False)).to(dtype=torch.int64, device=device) 
            elif self.stage == 'train': #80 sphere target views 
                index_target = torch.from_numpy(np.random.choice(np.arange(0, 80), size=self.cfg.num_target_views, 
                                                                replace=False)).to(dtype=torch.int64, device=device)
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
                                                                    replace=False)).to(dtype=torch.int64) #choice=nop.arange(0, 20), For test/val stages, you typically want deterministic/reproducible results, not random sampling
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
                index_target = torch.tensor([0,1,2,3,4,5])
        else: raise KeyError("Called dataset with wrong experiment argument ... ")

        return index_context, index_target
    
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views
    
    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views