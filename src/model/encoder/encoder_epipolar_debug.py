from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .epipolar.depth_predictor_monocular import DepthPredictorMonocular
from .epipolar.epipolar_transformer import EpipolarTransformer, EpipolarTransformerCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderEpipolarCfg:
    name: Literal["epipolar"]
    d_feature: int
    num_monocular_samples: int
    num_surfaces: int
    predict_opacity: bool
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    near_disparity: float
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    epipolar_transformer: EpipolarTransformerCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    use_epipolar_transformer: bool
    use_transmittance: bool


class EncoderEpipolar(Encoder[EncoderEpipolarCfg]):
    backbone: Backbone
    backbone_projection: nn.Sequential
    epipolar_transformer: EpipolarTransformer | None
    depth_predictor: DepthPredictorMonocular
    to_gaussians: nn.Sequential
    gaussian_adapter: GaussianAdapter
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderEpipolarCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        self.backbone_projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.backbone.d_out, cfg.d_feature),
        )
        if cfg.use_epipolar_transformer:
            self.epipolar_transformer = EpipolarTransformer(
                cfg.epipolar_transformer,
                cfg.d_feature,
            )
        else:
            self.epipolar_transformer = None
        self.depth_predictor = DepthPredictorMonocular(
            cfg.d_feature,
            cfg.num_monocular_samples,
            cfg.num_surfaces,
            cfg.use_transmittance,
        )
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        if cfg.predict_opacity:
            self.to_opacity = nn.Sequential(
                nn.ReLU(),
                nn.Linear(cfg.d_feature, 1),
                nn.Sigmoid(),
            )
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                cfg.d_feature,
                cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
            ),
        )
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, cfg.d_feature, 7, 1, 3),
            nn.ReLU(),
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # DEBUG: Print context camera information
        print(f'\n🔍 DEBUG: CONTEXT CAMERAS')
        print(f'  Batch size: {b}, Views: {v}, Image size: {h}x{w}')
        if "extrinsics" in context:
            context_positions = context["extrinsics"][:, :, :3, 3]  # Extract positions
            print(f'  Context camera positions:')
            for view_idx in range(min(v, 6)):  # Show first 6 views
                pos = context_positions[0, view_idx]
                print(f'    View {view_idx}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]')
            context_distances = torch.norm(context_positions, dim=-1)
            print(f'  Context distances from origin: [{context_distances.min():.2f}, {context_distances.max():.2f}]')

        # Encode the context images.
        features = self.backbone(context)
        features = rearrange(features, "b v c h w -> b v h w c")
        features = self.backbone_projection(features)
        features = rearrange(features, "b v h w c -> b v c h w")

        # Run the epipolar transformer.
        if self.cfg.use_epipolar_transformer:
            features, sampling = self.epipolar_transformer(
                features,
                context["extrinsics"],
                context["intrinsics"],
                context["near"],
                context["far"],
            )

        # Add the high-resolution skip connection.
        skip = rearrange(context["image"], "b v c h w -> (b v) c h w")
        skip = self.high_resolution_skip(skip)
        features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b, v=v)

        # Sample depths from the resulting features.
        features = rearrange(features, "b v c h w -> b v (h w) c")
        depths, densities = self.depth_predictor.forward(
            features,
            context["near"],
            context["far"],
            deterministic,
            1 if deterministic else self.cfg.gaussians_per_pixel,
        )

        # DEBUG: Print depth information
        print(f'🔍 DEBUG: DEPTH PREDICTION')
        print(f'  Depths shape: {depths.shape}')
        print(f'  Depth range: [{depths.min():.3f}, {depths.max():.3f}]')
        print(f'  Depth mean: {depths.mean():.3f}')
        print(f'  Near/Far bounds: [{context["near"].mean().item():.3f}, {context["far"].mean().item():.3f}]')

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            self.to_gaussians(features),
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
            (h, w),
        )

        # DEBUG: Print gaussian information AFTER gaussian_adapter
        print(f'🔍 DEBUG: GAUSSIAN GENERATION')
        print(f'  Gaussians means shape: {gaussians.means.shape}')
        print(f'  Gaussian positions:')
        means_flat = gaussians.means.view(-1, 3)
        print(f'    X range: [{means_flat[:, 0].min():.3f}, {means_flat[:, 0].max():.3f}]')
        print(f'    Y range: [{means_flat[:, 1].min():.3f}, {means_flat[:, 1].max():.3f}]')
        print(f'    Z range: [{means_flat[:, 2].min():.3f}, {means_flat[:, 2].max():.3f}]')
        distances_from_origin = torch.norm(means_flat, dim=-1)
        print(f'    Distance from origin: [{distances_from_origin.min():.3f}, {distances_from_origin.max():.3f}]')
        print(f'    Mean distance: {distances_from_origin.mean():.3f}')
        
        if hasattr(gaussians, 'scales'):
            scales_flat = gaussians.scales.view(-1, 3)
            print(f'  Gaussian scales:')
            print(f'    Scale range: [{scales_flat.min():.6f}, {scales_flat.max():.6f}]')
            print(f'    Mean scale: {scales_flat.mean():.6f}')
            print(f'    Scale std: {scales_flat.std():.6f}')
        
        if hasattr(gaussians, 'opacities'):
            opacities_flat = gaussians.opacities.view(-1)
            print(f'  Gaussian opacities:')
            print(f'    Opacity range: [{opacities_flat.min():.6f}, {opacities_flat.max():.6f}]')
            print(f'    Mean opacity: {opacities_flat.mean():.6f}')
            visible_count = (opacities_flat > 0.1).sum().item()
            total_count = opacities_flat.numel()
            print(f'    Visible gaussians (opacity > 0.1): {visible_count}/{total_count} ({100*visible_count/total_count:.1f}%)')

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            if self.cfg.use_epipolar_transformer:
                visualization_dump["sampling"] = sampling

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = (
            rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
            if self.cfg.predict_opacity
            else 1
        )

        final_gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        # DEBUG: Print final gaussian statistics
        print(f'🔍 DEBUG: FINAL GAUSSIANS')
        print(f'  Final means shape: {final_gaussians.means.shape}')
        print(f'  Final means range:')
        print(f'    X: [{final_gaussians.means[:, :, 0].min():.3f}, {final_gaussians.means[:, :, 0].max():.3f}]')
        print(f'    Y: [{final_gaussians.means[:, :, 1].min():.3f}, {final_gaussians.means[:, :, 1].max():.3f}]')
        print(f'    Z: [{final_gaussians.means[:, :, 2].min():.3f}, {final_gaussians.means[:, :, 2].max():.3f}]')
        final_distances = torch.norm(final_gaussians.means, dim=-1)
        print(f'  Final distances: [{final_distances.min():.3f}, {final_distances.max():.3f}] mean: {final_distances.mean():.3f}')
        print(f'  Final opacities: [{final_gaussians.opacities.min():.6f}, {final_gaussians.opacities.max():.6f}] mean: {final_gaussians.opacities.mean():.6f}')
        final_visible = (final_gaussians.opacities > 0.1).sum().item()
        final_total = final_gaussians.opacities.numel()
        print(f'  Final visible: {final_visible}/{final_total} ({100*final_visible/final_total:.1f}%)')
        print(f'🔍 DEBUG: END\n')

        return final_gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.epipolar_transformer.self_attention.patch_size
                * self.cfg.epipolar_transformer.downscale,
            )

            if self.cfg.apply_bounds_shim:
                _, _, _, h, w = batch["context"]["image"].shape
                near_disparity = self.cfg.near_disparity * min(h, w)
                batch = apply_bounds_shim(batch, near_disparity, 0.5)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return self.epipolar_transformer.epipolar_sampler