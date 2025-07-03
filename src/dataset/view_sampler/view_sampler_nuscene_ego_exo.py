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
    # Handle edge cases that cause "probabilities do not sum to 1"
    x = np.array(x, dtype=np.float64)  # Use higher precision
    
    # Check for invalid inputs
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print(f"Warning: Invalid values in softmax input: {x}")
        # Return uniform distribution as fallback
        return np.ones(len(x)) / len(x)
    
    # Prevent overflow by subtracting max
    x_max = np.max(x)
    if np.isinf(x_max) or np.isnan(x_max):
        return np.ones(len(x)) / len(x)
        
    e_x = np.exp(x - x_max)
    
    # Check for underflow (all zeros)
    sum_e_x = e_x.sum()
    if sum_e_x == 0 or np.isnan(sum_e_x) or np.isinf(sum_e_x):
        print(f"Warning: Invalid softmax sum: {sum_e_x}, returning uniform distribution")
        return np.ones(len(x)) / len(x)
    
    result = e_x / sum_e_x
    
    # Final check that probabilities sum to 1
    if abs(result.sum() - 1.0) > 1e-6:
        print(f"Warning: Probabilities don't sum to 1: {result.sum()}, normalizing")
        result = result / result.sum()
    
    return result

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
        num_available_context = extrinsics_context.shape[0]
        num_available_target = extrinsics_target.shape[0]
        
        print(f"ViewSampler DEBUG: Available cameras - context: {num_available_context}, target: {num_available_target}")
        print(f"ViewSampler DEBUG: Requested views - context: {self.cfg.num_context_views}, target: {self.cfg.num_target_views}")
        
        # Context view selection - NuScenes 6 cameras (same logic as SEED4D)
        if self.cfg.num_context_views < num_available_context:
            # Use SEED4D's camera ordering preference, but ensure indices are valid
            Choice = [i for i in [0, 1, 5, 3, 4, 2] if i < num_available_context]
            start = random.randint(0, len(Choice) - 1)
            selected_indices = list(islice(cycle(Choice), start, start + self.cfg.num_context_views))
            index_context = torch.tensor(selected_indices).to(dtype=torch.int64, device=device)
        else:
            # Use all available cameras
            index_context = torch.arange(min(self.cfg.num_context_views, num_available_context)).to(dtype=torch.int64, device=device)
        
        print(f"ViewSampler DEBUG: Selected context indices: {index_context}")
        
        # Target view selection with similarity-based sampling (same as SEED4D)
        # Convert to numpy for calculations but keep precision
        try:
            context_camera_directions = extrinsics_context[index_context, :3, 2].cpu().numpy().astype(np.float64)
            target_camera_directions = extrinsics_target[:, :3, 2].cpu().numpy().astype(np.float64)
            
            print(f"ViewSampler DEBUG: Context directions shape: {context_camera_directions.shape}")
            print(f"ViewSampler DEBUG: Target directions shape: {target_camera_directions.shape}")
            
            # Check for invalid values
            if np.any(np.isnan(context_camera_directions)) or np.any(np.isinf(context_camera_directions)):
                print("ViewSampler ERROR: Invalid context camera directions!")
                raise ValueError("Invalid context camera directions")
            if np.any(np.isnan(target_camera_directions)) or np.any(np.isinf(target_camera_directions)):
                print("ViewSampler ERROR: Invalid target camera directions!")
                raise ValueError("Invalid target camera directions")
            
            # Calculate similarity
            context_target_similarity = np.einsum("ij,lj->il", context_camera_directions, target_camera_directions)
            print(f"ViewSampler DEBUG: Similarity matrix shape: {context_target_similarity.shape}")
            
            # Apply temperature scaling to similarity scores
            target_sample_weight_map = np.array([softmax(temperature * row) for row in context_target_similarity])
            target_sample_weight = np.max(target_sample_weight_map, axis=0)
            target_sample_weight = softmax(temperature * target_sample_weight)
            
            print(f"ViewSampler DEBUG: Target weights shape: {target_sample_weight.shape}")
            print(f"ViewSampler DEBUG: Target weights sum: {target_sample_weight.sum()}")
            print(f"ViewSampler DEBUG: Target weights range: [{target_sample_weight.min():.6f}, {target_sample_weight.max():.6f}]")
            
        except Exception as e:
            print(f"ViewSampler ERROR in similarity calculation: {e}")
            # Fallback to uniform distribution
            target_sample_weight = np.ones(num_available_target) / num_available_target
            print("ViewSampler: Using uniform target sampling as fallback")
        
        # Stage-dependent target sampling (same limits as SEED4D)
        try:
            if self.stage == 'test' or self.stage == 'val':
                # Use predefined target views if specified
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                else:
                    # Sample from available target views
                    available_targets = min(num_available_target, len(target_sample_weight))
                    num_to_sample = min(self.cfg.num_target_views, available_targets)
                    
                    print(f"ViewSampler DEBUG: Sampling {num_to_sample} targets from {available_targets} available")
                    
                    # Ensure we have valid probabilities
                    weights = target_sample_weight[:available_targets]
                    if len(weights) == 0 or weights.sum() == 0:
                        weights = np.ones(available_targets) / available_targets
                    
                    index_target = torch.from_numpy(
                        np.random.choice(
                            np.arange(0, available_targets), 
                            size=num_to_sample,
                            replace=False, 
                            p=weights
                        )
                    ).to(dtype=torch.int64, device=device)
            
            # Training stage - sample from available views
            elif self.stage == 'train':
                available_targets = num_available_target
                num_to_sample = min(self.cfg.num_target_views, available_targets)
                
                # Ensure we have valid probabilities
                weights = target_sample_weight[:available_targets]
                if len(weights) == 0 or weights.sum() == 0:
                    weights = np.ones(available_targets) / available_targets
                
                index_target = torch.from_numpy(
                    np.random.choice(
                        np.arange(0, available_targets), 
                        size=num_to_sample,
                        replace=False, 
                        p=weights
                    )
                ).to(dtype=torch.int64, device=device)
            else:
                raise KeyError("Called dataset with wrong stage argument")
                
        except Exception as e:
            print(f"ViewSampler ERROR in target sampling: {e}")
            # Fallback to sequential sampling
            num_to_sample = min(self.cfg.num_target_views, num_available_target)
            index_target = torch.arange(num_to_sample).to(dtype=torch.int64, device=device)
            print(f"ViewSampler: Using sequential target sampling as fallback: {index_target}")
        
        print(f"ViewSampler DEBUG: Final context indices: {index_context}")
        print(f"ViewSampler DEBUG: Final target indices: {index_target}")
        
        # Return only 2 values (same as SEED4D ViewSamplerIOsplat)
        return index_context, index_target
    
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views
    
    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views