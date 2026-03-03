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
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Depth loss clipped to match ground truth range (0 to 65.535 meters)
        predicted_depth_clipped = torch.clamp(prediction.depth, min=0.0, max=65.535) #example depth=140 set to 65.535 max now to avoid having big loss there
        target_depth_clipped = torch.clamp(batch["target"]["depth"], min=0.0, max=65.535)
        delta = predicted_depth_clipped - target_depth_clipped
        #delta = prediction.depth - batch["target"]["depth"] # Original line without clipping
        
        return self.cfg.weight * torch.abs(delta).mean()