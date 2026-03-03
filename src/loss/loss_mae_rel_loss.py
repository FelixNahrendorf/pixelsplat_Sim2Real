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
        
        return pearson_loss_value