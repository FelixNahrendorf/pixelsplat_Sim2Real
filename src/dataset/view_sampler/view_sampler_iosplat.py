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
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@dataclass
class ViewSamplerIOsplatCfg:
    name: Literal["train_carla_view", "eval_carla_view"]
    num_context_views: int
    num_target_views: int
    augment: bool
    augment_p: float
    augment_mask_count: int
    context_views: list[int] | None
    target_views: list[int] | None
    input_context_resolution: int
    output_target_resolution: int


class ViewSamplerIOsplat(ViewSampler[ViewSamplerIOsplatCfg]):
    def sample(
        self,
        scene,
        extrinsics_context: Float[Tensor, "cview 4 4"],
        extrinsics_target: Float[Tensor, "tview 4 4"],
        experiment: str = "ego-exo",
        device: torch.device = torch.device("cpu"),
        use_nuscene_context: bool = False,
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        """Sample context and target view indices based on the experiment mode and stage.

        Context views are always drawn from the 6 ego-vehicle cameras (indices 0–5).
        Target views are sampled based on camera direction similarity to context views,
        or fixed during validation/testing.

        """
        # --- Context view selection ---
        if self.cfg.num_context_views < 6:
            # Cycle to avoid always picking the lowest-index cameras
            choices = [0, 1, 2, 3, 4, 5]
            start = random.randint(0, len(choices) - 1)
            index_context = torch.tensor(
                list(islice(cycle(choices), start, start + self.cfg.num_context_views)),
                dtype=torch.int64,
            )
        else:
            index_context = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device=device)

        # --- Similarity-weighted target sampling ---
        # Column 2 of the extrinsic matrix is the camera's forward (z) direction
        context_camera_directions = extrinsics_context[index_context, :, 2]
        target_camera_directions = extrinsics_target[:, :, 2]
        context_target_similarity = np.einsum("ij,lj->il", context_camera_directions, target_camera_directions)

        # Aggregate per-target similarity score using max over context views, then re-weight
        target_sample_weight_map = np.array([softmax(row) for row in context_target_similarity])
        target_sample_weight = softmax(np.max(target_sample_weight_map, axis=0))

        # --- Target view selection per experiment mode ---

        if experiment == "ego-exo-mixed":
            # ~~~ ego-exo-mixed ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.stage in ("test", "val"):
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                else:
                    assert self.cfg.num_target_views <= 20
                    index_target = torch.from_numpy(
                        np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, replace=False)
                    ).to(dtype=torch.int64)
            elif self.stage == "train":
                # 80 sphere targets + 18 SEED4D ego views = 98 total
                index_target = torch.from_numpy(
                    np.random.choice(np.arange(0, 98), size=self.cfg.num_target_views, replace=False)
                ).to(dtype=torch.int64)
            else:
                raise KeyError("Called dataset with wrong stage argument ... ")

        elif experiment == "ego-exo-mixed-domain":
            # ~~~ ego-exo-mixed-domain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # nuScenes ego views of are concatenated after SEED4D ego views, so context starts at index 6
            nuscene_context_indices = [6, 7, 8, 9, 10, 11]
            # 3*6 SEED4D ego views are appended to the 80 sphere targets, then 1*6 nuScenes ego views
            nuscene_target_indices = [98, 99, 100, 101, 102, 103]

            if use_nuscene_context:
                index_context = torch.tensor(nuscene_context_indices, dtype=torch.int64, device=device)

                if self.stage in ("test", "val"):
                    if self.cfg.target_views is not None:
                        assert len(self.cfg.target_views) == self.cfg.num_target_views
                        index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                    else:
                        # Tile the 6 nuScenes indices to fill num_target_views (default 20)
                        nuscene_target_indices_extended = (nuscene_target_indices * 3)[:18] + random.choices(
                            nuscene_target_indices, k=2
                        )
                        index_target = torch.tensor(nuscene_target_indices_extended, dtype=torch.int64, device=device)
                elif self.stage == "train":
                    index_target = torch.from_numpy(
                        np.random.choice(nuscene_target_indices, size=self.cfg.num_target_views, replace=False)
                    ).to(dtype=torch.int64, device=device)

            else:
                if self.cfg.num_context_views <= 6:
                    index_context = torch.arange(0, self.cfg.num_context_views, dtype=torch.int64, device=device)
                else:
                    index_context = torch.arange(0, 6, dtype=torch.int64, device=device)
                print("[SEED4D ONLY] Using SEED4D-only context")

                if self.stage in ("test", "val"):
                    if self.cfg.target_views is not None:
                        assert len(self.cfg.target_views) == self.cfg.num_target_views
                        index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                    else:
                        assert self.cfg.num_target_views <= 20
                        index_target = torch.from_numpy(
                            np.random.choice(np.arange(0, 20), size=self.cfg.num_target_views, replace=False)
                        ).to(dtype=torch.int64, device=device)
                elif self.stage == "train":
                    # 80 sphere targets + 18 SEED4D ego views = 98 total
                    index_target = torch.from_numpy(
                        np.random.choice(np.arange(0, 98), size=self.cfg.num_target_views, replace=False)
                    ).to(dtype=torch.int64, device=device)

        elif experiment == "ego-ego-nuscenes":
            # ~~~ ego-ego-nuscenes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Pure nuScenes: context and target share the same 6 camera indices
            nuscene_context_indices = [0, 1, 2, 3, 4, 5]
            nuscene_target_indices = [0, 1, 2, 3, 4, 5]

            index_context = torch.tensor(nuscene_context_indices, dtype=torch.int64, device=device)

            if self.stage in ("test", "val"):
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                else:
                    index_target = torch.tensor(nuscene_target_indices, dtype=torch.int64, device=device)
            elif self.stage == "train":
                index_target = torch.from_numpy(
                    np.random.choice(nuscene_target_indices, size=self.cfg.num_target_views, replace=False)
                ).to(dtype=torch.int64, device=device)

        elif experiment == "ego-exo-nuscenes":
            # ~~~ ego-exo-nuscenes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # nuScenes context (6 cameras) with exo sphere targets
            nuscene_context_indices = [0, 1, 2, 3, 4, 5]
            index_context = torch.tensor(nuscene_context_indices, dtype=torch.int64, device=device)

            if self.stage in ("test", "val"):
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                else:
                    assert self.cfg.num_target_views <= 20
                    n = len(extrinsics_target)
                    size = min(self.cfg.num_target_views, n)
                    index_target = torch.from_numpy(
                        np.random.choice(np.arange(0, n), size=size, replace=False)
                    ).to(dtype=torch.int64)
                    print("index_target ### test/val step ###", index_target)
            elif self.stage == "train":
                # 80 sphere target views
                index_target = torch.from_numpy(
                    np.random.choice(np.arange(0, 80), size=self.cfg.num_target_views, replace=False)
                ).to(dtype=torch.int64, device=device)

        elif experiment == "ego-exo-nuscenes-scene":
            # ~~~ ego-exo-nuscenes-scene ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # nuScenes context with a fixed BEV target (index 97) during eval
            nuscene_context_indices = [0, 1, 2, 3, 4, 5]
            index_context = torch.tensor(nuscene_context_indices, dtype=torch.int64, device=device)

            if self.stage in ("test", "val"):
                if self.cfg.target_views is not None:
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                else:
                    # Index 97 corresponds to the birds-eye-view (BEV) camera
                    index_target = torch.tensor([97], dtype=torch.int64, device=device)
            elif self.stage == "train":
                # 80 sphere target views
                index_target = torch.from_numpy(
                    np.random.choice(np.arange(0, 80), size=self.cfg.num_target_views, replace=False)
                ).to(dtype=torch.int64, device=device)

        elif experiment == "ego-exo":
            # ~~~ ego-exo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.stage in ("test", "val"):
                if self.cfg.target_views is not None:
                    assert len(self.cfg.target_views) == self.cfg.num_target_views
                    index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
                else:
                    assert self.cfg.num_target_views <= 20
                    n = len(extrinsics_target)
                    size = min(self.cfg.num_target_views, n)
                    index_target = torch.from_numpy(
                        np.random.choice(np.arange(0, n), size=size, replace=False)
                    ).to(dtype=torch.int64)
                    print("index_target ### test/val step ###", index_target)
            elif self.stage == "train":
                # Renormalize after slicing to the actual number of available targets
                n = len(extrinsics_target)
                w = target_sample_weight[:n]
                w = w / w.sum()
                index_target = torch.from_numpy(
                    np.random.choice(np.arange(0, n), size=self.cfg.num_target_views, replace=False, p=w)
                ).to(dtype=torch.int64)
                print("index_target ### train step ###", index_target)

        elif experiment == "ego-ego":
            # ~~~ ego-ego ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.cfg.target_views is not None:
                assert len(self.cfg.target_views) == self.cfg.num_target_views
                index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)
            else:
                # Fixed order for deterministic visual comparison across runs
                index_target = torch.tensor([0, 1, 2, 3, 4, 5])

        else:
            raise KeyError("Called dataset with wrong experiment argument ... ")

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views