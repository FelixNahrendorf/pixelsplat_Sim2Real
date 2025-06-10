### This code is modified from https://github.com/dcharatan/pixelsplat/tree/main

from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_iosplat import ViewSamplerIOsplat, ViewSamplerIOsplatCfg
from .view_sampler_nuscene import ViewSampler_NUSCENE, ViewSampler_NUSCENECfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "train_carla_view": ViewSamplerIOsplat,
    "eval_carla_view": ViewSamplerIOsplat,
    "train_nuscene_view": ViewSampler_NUSCENE,
    "eval_nuscene_view": ViewSampler_NUSCENE}

ViewSamplerCfg = ViewSamplerIOsplatCfg | ViewSampler_NUSCENECfg

def get_view_sampler(
    cfg: ViewSamplerCfg,  stage: Stage, step_tracker: StepTracker | None) -> ViewSampler[Any]:
    # here we will call the relevant view sampler
    print(f"\n\tView Sampler name --> {cfg.name} under stage = {stage} \n")
    return VIEW_SAMPLERS[cfg.name](
        cfg, stage,
        step_tracker)