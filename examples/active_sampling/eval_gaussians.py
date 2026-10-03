import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from gsplat.rendering import rasterization

from train_active_sampling import GaussianPointField, load_sparse_views
from examples.active_sampling.gaussian_parameter_contract import load_field


def _save_side_by_side(output_path: Path, gt: torch.Tensor, pred: torch.Tensor) -> None:
    """Save GT and prediction side-by-side for quick visual inspection."""
    gt_np = (gt.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    pred_np = (pred.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    canvas = torch.from_numpy(gt_np)
    pred_tensor = torch.from_numpy(pred_np)
    stacked = torch.cat([canvas, pred_tensor], dim=1).cpu().numpy()
    Image.fromarray(stacked).save(output_path)


def evaluate(
    gaussian_path: Path,
    data_dir: Path,
    image_subdir: str,
    output_dir: Path,
    device: torch.device,
    lpips_net: str,
    select_indices: Optional[List[int]] = None,
    holdout_stride: int = 0,
) -> Tuple[dict, List[dict]]:
    state = torch.load(gaussian_path, map_location=device)
    field = GaussianPointField(state["means"], state["colors"], device, scale_init=0.01)
    load_field(field, state, device)
    means, quats, scales, opacities, colors = (
        field.means, field.quats, field.scales, field.opacities, field.colors
    )

    cameras, images, _ = load_sparse_views(
        data_dir=data_dir,
        device=device,
        max_views=-1,
        target_size=-1,
        select_indices=select_indices,
        image_subdir=image_subdir,
    )
    evaluated_indices = (
        list(select_indices)
        if select_indices is not None
        else list(range(0, len(cameras), holdout_stride))
        if holdout_stride > 0
        else list(range(len(cameras)))
    )
    if select_indices is None and holdout_stride > 0:
        cameras = [cameras[index] for index in evaluated_indices]
        images = images[evaluated_indices]

    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type=lpips_net, normalize=False
    ).to(device)

    per_view: List[dict] = []
    for idx, camera in enumerate(cameras):
        gt_image = images[idx].to(device)
        width = gt_image.shape[-1]
        height = gt_image.shape[-2]
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=camera.viewmat.unsqueeze(0),
            Ks=camera.intrinsics.unsqueeze(0),
            width=width,
            height=height,
        )
        pred = rendered[0, ..., :3].permute(2, 0, 1).contiguous()

        gt_batch = gt_image.unsqueeze(0)
        pred_batch = pred.unsqueeze(0)
        psnr_val = psnr_metric(pred_batch, gt_batch).item()
        ssim_val = ssim_metric(pred_batch, gt_batch).item()
        lpips_val = lpips_metric(pred_batch, gt_batch).item()

        _save_side_by_side(
            render_dir / f"view_{idx:02d}.png",
            gt_image.cpu(),
            pred.cpu(),
        )

        per_view.append(
            {
                "index": evaluated_indices[idx],
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
            }
        )

    mean_stats = {
        "psnr": float(sum(item["psnr"] for item in per_view) / len(per_view)),
        "ssim": float(sum(item["ssim"] for item in per_view) / len(per_view)),
        "lpips": float(sum(item["lpips"] for item in per_view) / len(per_view)),
    }
    return mean_stats, per_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved Gaussian field.")
    parser.add_argument("--gaussian-path", type=Path, required=True, help="Path to gaussians.pt")
    parser.add_argument("--data-dir", type=Path, required=True, help="Scene directory (COLMAP format).")
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="images",
        help="Image subdirectory to evaluate on (e.g., images_8).",
    )
    parser.add_argument(
        "--select-indices",
        type=str,
        default="",
        help="Comma-separated list of image indices (sorted by filename) to evaluate, e.g., '8,16'.",
    )
    parser.add_argument(
        "--holdout-stride",
        type=int,
        default=8,
        help="Evaluate every N-th sorted image; LLFF uses 8. Ignored when --select-indices is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store metrics and renderings.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument(
        "--lpips-net",
        type=str,
        choices=["alex", "squeeze", "vgg"],
        default="squeeze",
        help="LPIPS backbone used by torchmetrics.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    select_indices: Optional[List[int]] = None
    if args.select_indices:
        try:
            select_indices = [
                int(part)
                for part in str(args.select_indices).split(",")
                if part.strip() != ""
            ]
        except ValueError as exc:  # noqa: BLE001
            raise ValueError(f"Failed to parse --select-indices: {args.select_indices}") from exc

    mean_stats, per_view = evaluate(
        gaussian_path=args.gaussian_path,
        data_dir=args.data_dir,
        image_subdir=args.image_subdir,
        output_dir=args.output_dir,
        device=device,
        lpips_net=args.lpips_net,
        select_indices=select_indices,
        holdout_stride=max(0, args.holdout_stride),
    )
    metrics = {
        "mean": mean_stats,
        "per_view": per_view,
        "eval_config": {
            "lpips_net": args.lpips_net,
            "holdout_stride": max(0, args.holdout_stride),
            "select_indices": select_indices,
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"Mean PSNR={mean_stats['psnr']:.3f}, SSIM={mean_stats['ssim']:.4f}, LPIPS={mean_stats['lpips']:.4f}"
    )


if __name__ == "__main__":
    main()
