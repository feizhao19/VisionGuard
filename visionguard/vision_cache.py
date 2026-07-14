"""Cache frozen vision-tower outputs so TTA does not re-encode the same image.

Across RL steps we still must re-run the language-model prefill: LayerNorm
updates change how visual tokens are projected into K/V. What we can safely
reuse is the continuous visual token embeddings produced by the frozen vision
encoder (+ projector for LLaVA; + optionally only the ViT for InstructBLIP).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn


def _match_batch(cached: torch.Tensor, batch_size: int) -> torch.Tensor:
    if cached.shape[0] == batch_size:
        return cached
    if cached.shape[0] != 1:
        raise ValueError(
            f"Cached visual features have batch {cached.shape[0]}, "
            f"cannot expand to batch {batch_size}."
        )
    return cached.expand(batch_size, *cached.shape[1:]).contiguous()


def encode_llava_image_features(
    model: nn.Module,
    pixel_values: torch.Tensor,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
) -> torch.Tensor:
    """Run LLaVA vision tower + projector once; return visual tokens ``[1, N, D]``."""
    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else model.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else model.config.vision_feature_select_strategy
    )
    with torch.no_grad():
        features = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
    return features.detach()


@contextmanager
def cached_llava_image_features(
    model: nn.Module,
    image_features: torch.Tensor,
) -> Iterator[None]:
    """Reuse precomputed LLaVA visual tokens inside ``forward`` / ``generate``.

    Keeps passing ``pixel_values`` so HuggingFace still merges image tokens into
    the prompt; only ``get_image_features`` is short-circuited.
    """
    original = model.get_image_features

    def _cached_get_image_features(
        pixel_values: torch.Tensor,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs,
    ):
        del vision_feature_layer, vision_feature_select_strategy, kwargs
        batch_size = pixel_values.shape[0]
        feats = _match_batch(image_features, batch_size)
        return feats.to(device=pixel_values.device, dtype=image_features.dtype)

    model.get_image_features = _cached_get_image_features
    try:
        yield
    finally:
        model.get_image_features = original


def encode_instructblip_vision_embeds(
    model: nn.Module,
    pixel_values: torch.Tensor,
    interpolate_pos_encoding: bool = False,
) -> torch.Tensor:
    """Run InstructBLIP vision encoder once; return ``[1, N, D]`` patch embeds.

    Q-Former still runs each call because it is conditioned on the text prompt
    (and on per-beam captions in the current TTA notebooks).
    """
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        embeds = vision_outputs.last_hidden_state
    return embeds.detach()


class _CachedVisionModelOutput:
    """Minimal stand-in for ``BaseModelOutputWithPooling`` / tuple indexing."""

    __slots__ = ("last_hidden_state",)

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.last_hidden_state
        raise IndexError(index)


@contextmanager
def cached_instructblip_vision_embeds(
    model: nn.Module,
    vision_embeds: torch.Tensor,
) -> Iterator[None]:
    """Reuse precomputed InstructBLIP ViT outputs inside ``forward`` / ``generate``."""
    vision_model = model.vision_model
    original_forward = vision_model.forward

    def _cached_forward(pixel_values: torch.Tensor, *args, **kwargs):
        del args, kwargs
        batch_size = pixel_values.shape[0]
        embeds = _match_batch(vision_embeds, batch_size)
        embeds = embeds.to(device=pixel_values.device, dtype=vision_embeds.dtype)
        return _CachedVisionModelOutput(embeds)

    vision_model.forward = _cached_forward
    try:
        yield
    finally:
        vision_model.forward = original_forward
