"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from locality_alignment.train_utils import load_checkpoint
from timm.layers import Mlp
from timm.models.vision_transformer import Block, VisionTransformer

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)


class TransformerProjector(nn.Module):
    """Transformer-based projector module, can initialize with MaskEmbed decoder weights"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_patch_tokens: int,
        num_prefix_tokens: int,
        output_size: int,
        num_layers: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        init_values: Optional[float] = 1e-6,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        checkpoint_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Input projection
        self.in_proj = nn.Linear(embed_dim, embed_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patch_tokens + num_prefix_tokens, embed_dim) * 0.02)

        # Prefix tokens
        self.num_prefix_tokens = num_prefix_tokens

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Output head
        self.head = nn.Linear(embed_dim, output_size)

        # Initialize from checkpoint
        self.checkpoint_name = checkpoint_name
        if checkpoint_name is not None:
            self.load_from_checkpoint(checkpoint_name)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply input projection
        x = self.in_proj(x)

        # Mask prefix tokens if transformer is MaskEmbed decoder
        if self.num_prefix_tokens > 0 and self.checkpoint_name is not None:
            x[:, : self.num_prefix_tokens] = 0

        # Generate predictions
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)

        return x

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load parameters from pre-trained MaskEmbed decoder checkpoint"""
        # Custom filter function
        def filter_fn(state_dict: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
            # Move encoder output head to decoder
            state_dict["decoder.in_proj.weight"] = state_dict.pop("encoder.head.weight")
            state_dict["decoder.in_proj.bias"] = state_dict.pop("encoder.head.bias")

            # Remove decoder output head keys
            del state_dict["decoder.head.weight"]
            del state_dict["decoder.head.bias"]

            # Remove encoder keys and rename decoder keys
            state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}

            return state_dict

        # Load state dict
        non_matching_keys = load_checkpoint(self, checkpoint_path, filter_fn=filter_fn, strict=False)
        logging.info(f"Loaded decoder from {checkpoint_path}, non-matching keys: {non_matching_keys}")

        # Zero-init output head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)


def truncated_transformer_projector(featurizer: VisionTransformer, llm_dim: int, num_layers: int) -> nn.Module:
    """Helper function to truncate a vision backbone and use final layers as a projector."""
    # Extract norm and last `num_layers` blocks from vision backbone
    adapter_blocks = featurizer.blocks[-num_layers:]
    featurizer.blocks = featurizer.blocks[:-num_layers]
    norm = featurizer.norm
    featurizer.norm = torch.nn.Identity()

    # Create projector with linear adapter layer
    linear = torch.nn.Linear(featurizer.embed_dim, llm_dim)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    projector = torch.nn.Sequential(*adapter_blocks, norm, linear)

    return projector
