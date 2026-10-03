"""Factory for the paper's SD 2.1 + depth-ControlNet generative prior."""

from typing import Literal, Optional

import torch
import torch.nn as nn


BackboneType = Literal["controlnet"]


def create_gddn_model(
    backbone: BackboneType = "controlnet",
    pretrained_path: Optional[str] = None,
    dropout_rate: float = 0.2,
    torch_dtype: Optional[torch.dtype] = None,
    default_guidance_scale: float = 1.0,
    enable_plucker_conditioning: bool = False,
    controlnet_condition_channels: int = 3,
) -> nn.Module:
    """Create the generative prior used by UTrustGS."""
    if backbone != "controlnet":
        raise ValueError("UTrustGS only supports the paper's controlnet backbone.")
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    from gddn_controlnet import GDDN_ControlNet

    return GDDN_ControlNet(
        pretrained_model_path=pretrained_path,
        dropout_rate=dropout_rate,
        torch_dtype=torch_dtype,
        use_depth_estimation=True,
        default_guidance_scale=default_guidance_scale,
        enable_plucker_conditioning=enable_plucker_conditioning,
        controlnet_condition_channels=controlnet_condition_channels,
    )
