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
class ViewSampler_NUSCENE_EGO_EXOCfg:
    name: Literal["train_nuscene_ego_exo_view", "eval_nuscene_ego_exo_view"]
    num_context_views: int
    num_target_views: int
    augment: bool
    augment_p: float
    augment_mask_count: int
    nuscene_version: str
    context_views: list[int] | None  # Fixed context views if specified
    target_views: list[int] | None   # Fixed target views if specified
    input_context_resolution: int
    output_target_resolution: int

class ViewSampler_NUSCENE_EGO_EXO(ViewSampler[ViewSampler_NUSCENE_EGO_EXOCfg]):
    def sample(
        self,
        scene,
        extrinsics_context: Float[Tensor, "cview 4 4"],
        extrinsics_target: Float[Tensor, "tview 4 4"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],   # indices for target views
    ]:
        """Sample context and target views for spatial novel view synthesis.
        Context views: NuScenes 6-camera setup (ego vehicle sensors)
        Target views: SEED4D spherical cameras for bird's eye view generation
        
        Uses the same sampling logic as SEED4D (ViewSamplerIOsplat) for compatibility.
        """
        temperature = 1
        
        # Context view selection - NuScenes 6 cameras (same logic as SEED4D)
        if self.cfg.num_context_views < 6:
            # Use SEED4D's camera ordering preference
            Choice = [0, 1, 5, 3, 4, 2]
            start = random.randint(0, len(Choice) - 1)
            index_context = torch.tensor(
                list(islice(cycle(Choice), start, start + self.cfg.num_context_views))
            ).to(dtype=torch.int64, device=device)
        else:
            # Use all 6 cameras with SEED4D ordering
            index_context = torch.from_numpy(np.array([0, 1, 5, 3, 4, 2])).to(dtype=torch.int64, device=device)
        
        # Target view selection with similarity-based sampling (same as SEED4D)
        context_camera_directions = extrinsics_context[index_context, :, 2]
        target_camera_directions = extrinsics_target[:, :, 2]
        context_target_similarity = np.einsum("ij,lj->il", context_camera_directions, target_camera_directions)
        
        # Apply temperature scaling to similarity scores
        target_sample_weight_map = np.array([softmax(temperature * row) for row in context_target_similarity])
        target_sample_weight = np.max(target_sample_weight_map, axis=0)
        target_sample_weight = softmax(temperature * target_sample_weight)
        
        # Stage-dependent target sampling (same limits as SEED4D)
        if self.stage == 'test' or self.stage == 'val':
            # Use predefined target views if specified
            if self.cfg.target_views is not None:
                assert len(self.cfg.target_views) == self.cfg.num_target_views
                index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
            else:
                # Sample from 20 spherical views (SEED4D test/val limit)
                assert self.cfg.num_target_views <= 20
                available_targets = min(20, len(target_sample_weight))
                index_target = torch.from_numpy(
                    np.random.choice(
                        np.arange(0, available_targets), 
                        size=self.cfg.num_target_views,
                        replace=False, 
                        p=target_sample_weight[:available_targets]
                    )
                ).to(dtype=torch.int64, device=device)
        
        # Training stage - sample from 80 views (SEED4D train limit)
        elif self.stage == 'train':
            assert self.cfg.num_target_views <= 80
            available_targets = min(80, len(target_sample_weight))
            index_target = torch.from_numpy(
                np.random.choice(
                    np.arange(0, available_targets), 
                    size=self.cfg.num_target_views,
                    replace=False, 
                    p=target_sample_weight[:available_targets]
                )
            ).to(dtype=torch.int64, device=device)
        else:
            raise KeyError("Called dataset with wrong stage argument")
        
        # Return only 2 values (same as SEED4D ViewSamplerIOsplat)
        return index_context, index_target
    
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views
    
    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views