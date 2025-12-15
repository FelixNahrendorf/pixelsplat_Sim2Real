from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMaeCfg:
    weight: float


@dataclass
class LossMaeCfgWrapper:
    mae: LossMaeCfg


class LossMae(Loss[LossMaeCfg, LossMaeCfgWrapper]):
    def pearson_loss(
        self, 
        depth_pred: Float[Tensor, "..."], 
        depth_gt: Float[Tensor, "..."]
    ) -> Float[Tensor, ""]:
        """Compute Pearson correlation loss (1 - correlation coefficient)."""
        src = depth_pred - depth_pred.mean()
        target = depth_gt - depth_gt.mean()
        src = src / (src.std() + 1e-6)
        target = target / (target.std() + 1e-6)
        co = (src * target).mean()
        assert not torch.any(torch.isnan(co)), "NaN detected in Pearson correlation"
        return 1 - co
    
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Use the flag from batch - it knows the actual data source!
        #dataset_change = batch.get("dataset_change", False)
        
        # Convert Tensor to boolean
        #if isinstance(dataset_change, torch.Tensor):
        #    is_nuscene_batch = dataset_change.any().item()
        #else:
        #    is_nuscene_batch = bool(dataset_change)

        # Skip depth loss for nuScenes
        #if is_nuscene_batch:
        #    print(f"[LOSS DEBUG] [Step {global_step}] ----------- Skipping depth loss for nuScenes -----------------")
        #    return torch.tensor(0.0, device=prediction.depth.device, dtype=prediction.depth.dtype)

        # Depth loss clipped to match ground truth range (0 to 65.535 meters)
        #predicted_depth_clipped = torch.clamp(prediction.depth, min=0.0, max=65.535) #example depth=140 set to 65.535 max now to vaoid having big loss there
        #target_depth_clipped = torch.clamp(batch["target"]["depth"], min=0.0, max=65.535)
        #delta = predicted_depth_clipped - target_depth_clipped
        #delta = prediction.depth - batch["target"]["depth"] # Original line without clipping


        # Depth loss clipped to match ground truth range (0 to 65.535 meters)
        predicted_depth_clipped = torch.clamp(prediction.depth, min=0.0, max=65.535)
        target_depth_clipped = torch.clamp(batch["target"]["depth"], min=0.0, max=65.535)
        
        # Pearson loss (default)
        pearson_loss_value = self.cfg.weight * self.pearson_loss(
            predicted_depth_clipped, 
            target_depth_clipped
        )
        
        # Optionally add MAE loss
        #if self.cfg.use_mae:
        #    delta = predicted_depth_clipped - target_depth_clipped
        #    mae_loss = self.cfg.mae_weight * torch.abs(delta).mean()
        #    total_loss = pearson_loss_value + mae_loss
        #    return total_loss

        # # MAE loss (commented out - now optional)
        # delta = predicted_depth_clipped - target_depth_clipped
        # mae_loss = self.cfg.weight * torch.abs(delta).mean()
        # return mae_loss
        
        #return self.cfg.weight * torch.abs(delta).mean()
        
        return pearson_loss_value