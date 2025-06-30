from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, compute_depth_mse
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
import numpy as np
import json
import os 
import time

os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int

@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        
        # Run the model.
        gaussians = self.encoder(batch["context"], self.global_step, False)
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if self.global_rank == 0:
            print(
                f"train step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss
    
    def create_concatenated_image(self, scene_path, reference_images, color_images, target_images, depth_images):
        """Create a concatenated image with four horizontal rows stacked vertically."""
        try:
            # Convert tensors to numpy arrays and ensure they're in the right format
            def tensor_to_image_array(tensor_list):
                images = []
                for i, tensor in enumerate(tensor_list):
                    if isinstance(tensor, torch.Tensor):
                        # Convert to numpy and ensure proper format
                        img = tensor.detach().cpu().numpy()
                        
                        # Handle different tensor shapes
                        if len(img.shape) == 3:
                            if img.shape[0] in [1, 3, 4]:  # Channel first (C, H, W)
                                img = np.transpose(img, (1, 2, 0))
                            # If already (H, W, C), keep as is
                        elif len(img.shape) == 2:  # Grayscale (H, W)
                            img = np.stack([img] * 3, axis=2)  # Convert to RGB
                        
                        # Debug: Check for problematic values
                        if img.max() > 1.1 or img.min() < -0.1:
                            print(f"Debug - WARNING: Image {i} has unusual range [{img.min():.3f}, {img.max():.3f}]")
                        
                        # Ensure range [0, 1] - be more robust about range detection
                        if img.dtype == np.uint8:
                            img = img.astype(np.float32) / 255.0
                        elif img.max() > 1.0:
                            # Normalize values > 1.0 back to [0,1] range
                            img = np.clip(img, 0.0, 1.0)
                            print(f"Debug - Clipped image {i} to [0,1] range")
                        
                        # Ensure 3 channels for RGB
                        if len(img.shape) == 3 and img.shape[2] == 1:
                            img = np.repeat(img, 3, axis=2)
                        elif len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]  # Remove alpha channel if present
                        
                        # Final safety clip
                        img = np.clip(img, 0.0, 1.0)
                        
                        images.append(img)
                    else:
                        print(f"Warning: Non-tensor item in image list: {type(tensor)}")
                return images
            
            # Helper function to create a black image with white text
            def create_empty_image(target_h, target_w, text):
                img = np.zeros((target_h, target_w, 3), dtype=np.float32)
                try:
                    import cv2
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = min(target_h, target_w) / 400.0
                    color = (1.0, 1.0, 1.0)  # White color
                    thickness = max(1, int(font_scale * 2))
                    
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_x = (target_w - text_size[0]) // 2
                    text_y = (target_h + text_size[1]) // 2
                    
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
                except ImportError:
                    pass
                return img
            
            # Fixed padding function
            def pad_image_list(img_list, target_count, target_h, target_w, list_name):
                original_count = len(img_list)
                print(f"Debug - {list_name}: original={original_count}, target={target_count}")
                
                if len(img_list) == 0:
                    # Create empty images with text
                    empty_img = create_empty_image(target_h, target_w, f"{list_name} empty")
                    result = [empty_img.copy() for _ in range(target_count)]
                    print(f"Debug - Created {len(result)} empty images for {list_name}")
                    return result
                elif len(img_list) < target_count:
                    # Repeat the last image to fill the gap
                    last_img = img_list[-1].copy()
                    padding_needed = target_count - len(img_list)
                    padded_images = [last_img.copy() for _ in range(padding_needed)]
                    img_list.extend(padded_images)
                    print(f"Debug - Padded {list_name} from {original_count} to {len(img_list)} (added {padding_needed} copies)")
                    return img_list
                elif len(img_list) > target_count:
                    # Truncate to target count
                    img_list = img_list[:target_count]
                    print(f"Debug - Truncated {list_name} from {original_count} to {len(img_list)}")
                    return img_list
                else:
                    print(f"Debug - {list_name} already correct size: {len(img_list)}")
                    return img_list
            
            # Debug: Print tensor info before conversion
            print(f"Debug - Reference images: {len(reference_images)}")
            print(f"Debug - Color images: {len(color_images)}")
            print(f"Debug - Target images: {len(target_images)}")
            print(f"Debug - Depth images: {len(depth_images)}")
            
            if color_images:
                sample_color = color_images[0]
                if isinstance(sample_color, torch.Tensor):
                    print(f"Debug - Color tensor shape: {sample_color.shape}, dtype: {sample_color.dtype}, range: [{sample_color.min():.3f}, {sample_color.max():.3f}]")
            
            # Convert all image sets to numpy arrays
            ref_arrays = tensor_to_image_array(reference_images)
            color_arrays = tensor_to_image_array(color_images)
            target_arrays = tensor_to_image_array(target_images)
            depth_arrays = tensor_to_image_array(depth_images)
            
            print(f"Debug - After conversion - ref: {len(ref_arrays)}, color: {len(color_arrays)}, target: {len(target_arrays)}, depth: {len(depth_arrays)}")
            
            # Check if any arrays are empty
            if not (ref_arrays or color_arrays or target_arrays or depth_arrays):
                print(f"Warning: All image arrays are empty for scene {scene_path}")
                return
            
            # Find the maximum number of images across all types
            max_images = max(
                len(ref_arrays) if ref_arrays else 0,
                len(color_arrays) if color_arrays else 0,
                len(target_arrays) if target_arrays else 0,
                len(depth_arrays) if depth_arrays else 0
            )
            
            print(f"Debug - Max images: {max_images}")
            
            # Get dimensions - use the first available image from any array
            sample_img = None
            for img_list, name in [(ref_arrays, "ref"), (color_arrays, "color"), (target_arrays, "target"), (depth_arrays, "depth")]:
                if img_list:
                    sample_img = img_list[0]
                    print(f"Debug - Using {name} for dimensions: {sample_img.shape}")
                    break
            
            if sample_img is None:
                print(f"Warning: No valid images found for scene {scene_path}")
                return
                
            target_h, target_w = sample_img.shape[:2]
            
            # Resize all images to match target dimensions
            def resize_images(img_list, target_h, target_w, list_name):
                resized = []
                for i, img in enumerate(img_list):
                    if img.shape[:2] != (target_h, target_w):
                        try:
                            import cv2
                            img = cv2.resize(img, (target_w, target_h))
                            print(f"Debug - Resized {list_name}[{i}] to ({target_h}, {target_w})")
                        except ImportError:
                            try:
                                from scipy.ndimage import zoom
                                zoom_factors = (target_h / img.shape[0], target_w / img.shape[1], 1)
                                img = zoom(img, zoom_factors, order=1)
                                print(f"Debug - Zoom resized {list_name}[{i}] to ({target_h}, {target_w})")
                            except ImportError:
                                print(f"Warning: Cannot resize {list_name}[{i}] - no cv2 or scipy available")
                    resized.append(img)
                return resized
            
            # Resize all images first
            if ref_arrays:
                ref_arrays = resize_images(ref_arrays, target_h, target_w, "reference")
            if color_arrays:
                color_arrays = resize_images(color_arrays, target_h, target_w, "color")
            if target_arrays:
                target_arrays = resize_images(target_arrays, target_h, target_w, "target")
            if depth_arrays:
                depth_arrays = resize_images(depth_arrays, target_h, target_w, "depth")
            
            # Ensure all arrays have the same number of images using FIXED padding
            ref_arrays = pad_image_list(ref_arrays, max_images, target_h, target_w, "reference")
            color_arrays = pad_image_list(color_arrays, max_images, target_h, target_w, "color")
            target_arrays = pad_image_list(target_arrays, max_images, target_h, target_w, "target")
            depth_arrays = pad_image_list(depth_arrays, max_images, target_h, target_w, "depth")
            
            # Create horizontal concatenations
            try:
                ref_row = np.concatenate(ref_arrays, axis=1) if ref_arrays else np.zeros((target_h, target_w, 3))
                color_row = np.concatenate(color_arrays, axis=1) if color_arrays else np.zeros((target_h, target_w, 3))
                target_row = np.concatenate(target_arrays, axis=1) if target_arrays else np.zeros((target_h, target_w, 3))
                depth_row = np.concatenate(depth_arrays, axis=1) if depth_arrays else np.zeros((target_h, target_w, 3))
                
                print(f"Debug - Row shapes: ref{ref_row.shape}, color{color_row.shape}, target{target_row.shape}, depth{depth_row.shape}")
                
                # Stack vertically
                final_image = np.concatenate([ref_row, color_row, target_row, depth_row], axis=0)
                
                print(f"Debug - Final image shape: {final_image.shape}, range: [{final_image.min():.3f}, {final_image.max():.3f}]")
                
                # Convert back to tensor and save
                final_tensor = torch.from_numpy(final_image).permute(2, 0, 1).float()
                
                # Save the concatenated image
                concat_path = scene_path / "concatenated_view.png"
                save_image(final_tensor, concat_path)
                print(f"Saved concatenated image to {concat_path}")
                
            except Exception as e:
                print(f"Error during concatenation for {scene_path}: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error creating concatenated image for {scene_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        start_time = time.time()
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,  #RESTORED: Use deterministic=True for consistent test results - EDIT: # DEBUG: changed to False for testing
            )
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode="depth",
            )
        compute_time = time.time()-start_time
        (scene,) = (batch["scene"][0] + "_" + str(batch_idx),)
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        depth_prop = output.depth[0].unsqueeze(1)
        rgb_gt = batch["target"]["image"][0]
        depth_gt = batch["target"]["depth"][0]
        
        # Get reference (context) images
        reference_images = batch["context"]["image"][0]

        # Lists to store images for concatenation
        saved_reference_images = []
        saved_color_images = []
        saved_target_images = []
        saved_depth_images = []

        # Save images.
        if self.test_cfg.save_image:
            # Save rendered color images
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}.png")
                saved_color_images.append(color)
            
            # Save rendered depth images
            for index, depth_map in zip(batch["target"]["index"][0], depth_prop):
                save_image(depth_map.squeeze(0)/60, path / scene / f"depth/{index:0>6}.png")
                saved_depth_images.append(depth_map.squeeze(0)/60)
            
            # Save reference (context) images
            for index, reference_img in zip(batch["context"]["index"][0], reference_images):
                save_image(reference_img, path / scene / f"reference/{index:0>6}.png")
                saved_reference_images.append(reference_img)
            
            # Save target (ground truth) images
            for index, target_img in zip(batch["target"]["index"][0], rgb_gt):
                save_image(target_img, path / scene / f"target/{index:0>6}.png")
                saved_target_images.append(target_img)
            
            # Create concatenated image
            scene_path = path / scene
            self.create_concatenated_image(
                scene_path,
                saved_reference_images,
                saved_color_images, 
                saved_target_images,
                saved_depth_images
            )
        
        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []
            if f"drmse" not in self.test_step_outputs:
                self.test_step_outputs[f"drmse"] = []
            if f"compute_time" not in self.test_step_outputs:
                self.test_step_outputs[f"compute_time"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"compute_time"].append(compute_time)
            self.test_step_outputs[f"drmse"].append(
                torch.sqrt(compute_depth_mse(depth_gt.clamp(min=0.0, max=60.0),
                                            depth_prop.clamp(min=0.0, max=60.0), 
                                            output_color=rgb.clamp(min=0.0, max=1.0))).item())

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            # Note: benchmarker.clear_history() method may not be implemented
            # self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_probabilistic = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False, 
        )
        output_probabilistic = self.decoder.forward(
            gaussians_probabilistic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_probabilistic = output_probabilistic.color[0]
        gaussians_deterministic = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=True,
        )
        output_deterministic = self.decoder.forward(
            gaussians_deterministic,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_deterministic = output_deterministic.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("deterministic", "probabilistic"), (rgb_deterministic, rgb_probabilistic)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_probabilistic), "Target (Probabilistic)"),
            add_label(vcat(*rgb_deterministic), "Target (Deterministic)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = vcat(
            hcat(
                *render_projections(
                    gaussians_probabilistic,
                    256,
                    extra_label="(Probabilistic)",
                )[0]
            ),
            hcat(
                *render_projections(
                    gaussians_deterministic, 256, extra_label="(Deterministic)"
                )[0]
            ),
            align="left",
        )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        output_det = self.decoder.forward(
            gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_det = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(images_prob, images_det)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }