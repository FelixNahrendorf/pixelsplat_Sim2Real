from functools import cache

import torch
from einops import reduce, rearrange
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor

### Absolute Depth metrics
@torch.no_grad()
def compute_depth_mse(
    ground_truth: Float[Tensor, "batch 1 height width"],
    predicted: Float[Tensor, "batch 1 height width"],
    output_color: Float[Tensor, "batch c height width"]):
    # # we calculate depth error only over pixels different from background
    background_color = torch.Tensor([1.0, 1.0, 1.0]).to(device=predicted.device)
    valid_mask = torch.Tensor(rearrange(output_color, "b c h w -> b h w c") != background_color).to(dtype=float)
    valid_mask = rearrange(valid_mask.sum(-1)/3.0, "b h w -> b 1 h w")
    ground_truth = ground_truth.to(device=predicted.device)
    # Depth loss clipped to match ground truth range (0 to 65.535 meters)
    predicted_depth_clipped = torch.clamp(predicted, min=0.0, max=65.535)
    ground_truth_depth_clipped = torch.clamp(ground_truth, min=0.0, max=65.535)
    mse_loss =  torch.nn.MSELoss(reduction='none')(predicted_depth_clipped, ground_truth_depth_clipped)
    # mse_loss = (mse_loss * valid_mask.float()).sum()
    return (mse_loss * valid_mask.float()).sum() / valid_mask.sum()

###Relative Depth Metrics
'''@torch.no_grad()
def compute_depth_mse(
    ground_truth: Float[Tensor, "batch 1 height width"],
    predicted: Float[Tensor, "batch 1 height width"],
    output_color: Float[Tensor, "batch c height width"]):
    """
    Compute Pearson loss for depth (1 - correlation), replacing MSE.
    Returns mean Pearson loss as scalar (lower is better, range 0 to 2).
    Matches the Pearson loss used in training.
    """
    # we calculate depth error only over pixels different from background
    background_color = torch.Tensor([1.0, 1.0, 1.0]).to(device=predicted.device)
    valid_mask = torch.Tensor(rearrange(output_color, "b c h w -> b h w c") != background_color).to(dtype=float)
    valid_mask = rearrange(valid_mask.sum(-1)/3.0, "b h w -> b 1 h w") > 0.5
    
    ground_truth = ground_truth.to(device=predicted.device)
    
    # Clip to valid depth range (same as training loss)
    predicted_clipped = torch.clamp(predicted, min=0.0, max=65.535)
    ground_truth_clipped = torch.clamp(ground_truth, min=0.0, max=65.535)
    
    # Compute per-batch Pearson correlation
    correlations = []
    for i in range(predicted.shape[0]):
        # Get valid pixels for this batch item
        pred_i = predicted_clipped[i][valid_mask[i]]
        gt_i = ground_truth_clipped[i][valid_mask[i]]
        
        if len(pred_i) > 1:  # Need at least 2 points for correlation
            # Center the data
            pred_centered = pred_i - pred_i.mean()
            gt_centered = gt_i - gt_i.mean()
            
            # Normalize by standard deviation
            pred_normalized = pred_centered / (pred_centered.std() + 1e-6)
            gt_normalized = gt_centered / (gt_centered.std() + 1e-6)
            
            # Compute correlation coefficient
            corr = (pred_normalized * gt_normalized).mean()
            correlations.append(corr)
        else:
            # Not enough valid pixels - return worst case
            correlations.append(torch.tensor(-1.0, device=predicted.device))
    
    # Compute Pearson LOSS (1 - correlation) - matches training loss
    # correlation = 1.0 → loss = 0.0 (perfect)
    # correlation = 0.0 → loss = 1.0 (no correlation)
    # correlation = -1.0 → loss = 2.0 (inverse correlation)
    pearson_loss = 1.0 - torch.stack(correlations).mean()
    
    return pearson_loss'''


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)