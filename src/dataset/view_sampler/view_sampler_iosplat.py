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
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Sample target views depending on the training
            versus testing modes (target views are kept constant during inference)
            Context views are always set to conditional 6 images on Ego Vehicle
        """
        temperature = 1
        if self.cfg.num_context_views<6:
            Choice = [0, 1, 5, 3, 4, 2]
            start = random.randint(0, len(Choice) - 1)
            index_context = torch.tensor(list(islice(cycle(Choice), start, start + self.cfg.num_context_views))).to(dtype=torch.int64)
        else:
            index_context = torch.from_numpy(np.array([0, 1, 5, 3, 4, 2])).to(dtype=torch.int64, device=device)
            # index_context = torch.arange(0, 6, dtype=torch.int64, device=device)
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
        if self.stage=='test' or self.stage=='val':
            # If the target views are not None, then use them
            if self.cfg.target_views is not None: 
                assert len(self.cfg.target_views) == self.cfg.num_target_views
                index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
            # If not, then randomly select them
            else: # perform sanity check first and then create random indexes
                print('target_sample_weight:', target_sample_weight)
                print('target_sample_weight_string:', str(target_sample_weight))
                assert self.cfg.num_target_views<=20 
                index_target = torch.from_numpy(np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, 
                                                                 replace=False, p=target_sample_weight)).to(dtype=torch.int64)
        # Otherwise (training) will sample them randomly from 80 views
        elif self.stage == 'train':
            assert self.cfg.num_target_views<=80
            # #
            index_target = torch.from_numpy(np.random.choice(np.arange(0, 80), size=self.cfg.num_target_views, 
                                                             replace=False, p=target_sample_weight)).to(dtype=torch.int64)
        else: raise KeyError("Called dataset with wrong stage argument ... ")
        
        return index_context, index_target
    
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views
    
    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
