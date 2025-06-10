from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    background_color: list[float]
