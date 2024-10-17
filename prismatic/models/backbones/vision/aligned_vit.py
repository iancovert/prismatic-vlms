"""
aligned_vit.py

Vision transformers that have undergone locality alignment using MaskEmbed.
"""

import logging
from abc import ABC
from functools import partial
from typing import Callable

import timm
import torch
from locality_alignment import load_checkpoint_auto
from locality_alignment.backbones.svit import StableBlock
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import LetterboxPad, TimmViTBackbone, patch_featurizer_forward

# Registry =>> Supported Vision Backbones (from local checkpoint weights)
ALIGNED_BACKBONES = {
    # IN1k backbones
    "in1k-vit-l-aligned": ("svit_large_patch16_224", "vision_checkpoints/in1k-vit-l-maskembed"),
    # CLIP backbones
    "clip-vit-b-aligned": ("svit_base_patch16_clip_quickgelu_224", "vision_checkpoints/clip-vit-b-maskembed"),
    "clip-vit-l-aligned": ("svit_large_patch14_clip_quickgelu_224", "vision_checkpoints/clip-vit-l-maskembed"),
    "clip-vit-l-336px-aligned": ("svit_large_patch14_clip_quickgelu_336", "vision_checkpoints/clip-vit-l-336px-maskembed"),
    # SigLIP backbones
    "siglip-vit-b-aligned": ("svit_base_patch16_siglip_224", "vision_checkpoints/siglip-vit-b-maskembed"),
    "siglip-vit-so400m-aligned": ("svit_so400m_patch14_siglip_224", "vision_checkpoints/siglip-vit-so400m-maskembed"),
    "siglip-vit-so400m-384px-aligned": ("svit_so400m_patch14_siglip_384", "vision_checkpoints/siglip-vit-so400m-384px-maskembed"),
}


class AlignedViTBackbone(TimmViTBackbone, ABC):
    """Similar to TimmViTBackbone, but with custom checkpoint loading and forward for locality-aligned models"""

    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super(TimmViTBackbone, self).__init__(
            vision_backbone_id, image_resize_strategy, default_image_size=default_image_size
        )
        self.timm_path_or_url, self.checkpoint_name = ALIGNED_BACKBONES[vision_backbone_id]
        self.dtype = torch.bfloat16

        # Initialize Featurizer (ViT) using TIMM
        self.featurizer: VisionTransformer = timm.create_model(
            self.timm_path_or_url, pretrained=False, num_classes=0, img_size=self.default_image_size
        )
        self.featurizer.eval()

        # Load checkpoint weights
        non_matching_keys = load_checkpoint_auto(self.featurizer, self.checkpoint_name, strict=False)
        logging.info(f"Loaded vision backbone from {self.checkpoint_name}, non-matching keys: {non_matching_keys}")

        # Monkey-Patch the `forward()` function of the featurizer to ensure FSDP-compatibility
        #   => Note: for aligned models, we return the last layer's output and preserve prefix tokens
        patch_featurizer_forward(self.featurizer, skip_layers=0, prune_norm=False, trim_prefix=False)

        # Validation =>> for now, this class *only* supports TIMM Vision Transformers (but can be extended!)
        assert isinstance(self.featurizer, VisionTransformer), (
            "Featurizer is not a TIMM VisionTransformer; if you would like to support a new visual representation, "
            "file an issue or implement the requisite logic (see `prismatic/models/backbones/vision/base_vision.py`)!"
        )

        # Get Config =>> Note :: Override default image size to ensure correct image transform
        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        default_image_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        # Fix =>> SigLIP & IN1K default transforms resize to *larger* than `self.default_image_size` (crops image)!
        if "siglip" in self.timm_path_or_url or "in1k" in self.timm_path_or_url:
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)
            default_image_transform = Compose(
                [
                    Resize(self.default_image_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert isinstance(default_image_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_image_transform.transforms[0].interpolation),
                    *default_image_transform.transforms[1:],
                ]
            )

        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_image_transform

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_image_transform, Compose), "Unexpected `default_image_transform`!"
            assert "mean" in self.data_cfg, "TIMM `data_cfg` missing image normalization mean!"

            # Compute Padding Fill Value (rescaled normalization mean if applicable)
            fill = tuple([int(x * 255) for x in self.data_cfg["mean"]])

            # Build New Transform
            self.image_transform = Compose([LetterboxPad(fill), *default_image_transform.transforms])

        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        stable_transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={StableBlock})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy, stable_transformer_block_policy])
