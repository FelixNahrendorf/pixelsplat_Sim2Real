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
        delta = prediction.depth - batch["target"]["depth"]
        return self.cfg.weight * torch.abs(delta).mean()
