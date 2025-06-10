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
class ViewSampler_NUSCENECfg:
    name: Literal["train_nuscene_view", "eval_nuscene_view"]
    num_context_views: int
    num_target_views: int
    augment: bool
    augment_p: float
    augment_mask_count: int
    nuscene_td: int
    nuscene_render_k: int            # # determines the index of render view
    nuscene_version: str
    context_views: list[int] | None  # # context views are standing for reference views in Paper
    target_views: list[int] | None
    input_context_resolution: int
    output_target_resolution: int

class ViewSampler_NUSCENE(ViewSampler[ViewSampler_NUSCENECfg]):
    def sample(
        self,
        extrinsics: Float[Tensor, "seq view 4 4"], stage="test",
        # # # extrinsics \in [self.frame_seq_size, v, 4, 4] with v=6 for nuScene dataset 
        device: torch.device = torch.device("cpu")) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"], int, int # indices for target views
        ]:
        """Sample target and rendering views depending on the training
            versus testing modes (target views are kept constant during inference)
            Context views are always set to conditional 6 images on Ego Vehicle
        """
        seq, view, _, _ = extrinsics.shape
        self.all_frames = [frame for frame in range(0, 2 * self.cfg.nuscene_td + 1)]
        self.reference_frame = 0                           # # t 0 frame is the reference  
        self.target_frame = self.cfg.nuscene_render_k      # # t k frame is the render
        #######################################################################
        self.reference_view_count = view
        self.target_view_count = view 
        #######################################################################
        index_context = torch.arange(0, self.reference_view_count, dtype=torch.int64, device=device)
        index_target = torch.from_numpy(np.random.choice(np.arange(0, self.target_view_count), 
                                        size=self.cfg.num_target_views, replace=False)).to(dtype=torch.int64)
        if stage == "test":
            index_target = torch.arange(0, self.num_target_views, dtype=torch.int64, device=device)
        return index_context, index_target, self.reference_frame, self.target_frame
    
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views
    
    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views