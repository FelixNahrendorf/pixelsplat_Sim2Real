### This code is copied from https://github.com/dcharatan/pixelsplat/tree/main

from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_carla import Dataset_CARLA, Dataset_CARLACfg
from .dataset_nuScene import Dataset_NUSCENE, Dataset_NUSCENECfg
from .dataset_seed4d import Dataset_SEED4D, Dataset_SEED4DCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "carla": Dataset_CARLA,
    "nuscene": Dataset_NUSCENE,
    "seed4d": Dataset_SEED4D}

DatasetCfg = Dataset_CARLACfg | Dataset_NUSCENECfg | Dataset_SEED4DCfg

def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    # In our case views samplers could be different dependent on whether 
    # we are performing training vs. evaluation
    if cfg.name == "carla":
        if stage == 'train':        
            view_sampler = get_view_sampler(
                cfg.train_view_sampler, stage, step_tracker)
        elif stage == 'test' or stage == 'val':
            view_sampler = get_view_sampler(
                cfg.eval_view_sampler, stage, step_tracker)
    elif cfg.name == "nuscene":
        # # # as nuScene contains train&val sets together
        if stage == 'train' or stage == 'val':        
            view_sampler = get_view_sampler(
                cfg.train_view_sampler, stage, step_tracker)
        elif stage == 'test':
            view_sampler = get_view_sampler(
                cfg.eval_view_sampler, stage, step_tracker)
    elif cfg.name == "seed4d":
        if stage == 'train':        
            view_sampler = get_view_sampler(
                cfg.train_view_sampler, stage, step_tracker)
        elif stage == 'test' or stage == 'val':
            view_sampler = get_view_sampler(
                cfg.eval_view_sampler, stage, step_tracker)
    return DATASETS[cfg.name](cfg, stage, view_sampler)