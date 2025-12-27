"""
Utility helpers to export weight heatmaps from HY-WorldPlay double-stream blocks.

Defaults target blocks 8, 27, and 46 and layers listed in DEFAULT_LAYER_ORDER.
MLP layers are split into separate fc1 / fc2 heatmaps.
"""
import logging
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hyvideo.models.transformers.modules.mlp_layers import MLP
from hyvideo.models.transformers.modules.modulate_layers import ModulateDiT

logger = logging.getLogger(__name__)

# Layers to visualize inside each MMDoubleStreamBlock.
DEFAULT_LAYER_ORDER: Sequence[str] = (
    "img_attn_k",
    "img_attn_proj",
    "img_attn_prope_proj",
    "img_attn_q",
    "img_mlp",
    "img_mod",
    "txt_attn_k",
    "txt_attn_proj",
    "txt_attn_q",
    "txt_attn_v",
    "txt_mlp",
    "txt_mod",
)

# Explicitly note where we combine multiple weights into a single heatmap.
# Currently unused because MLPs are rendered separately.
LAYER_COMBINE_NOTES: Mapping[str, str] = {}

DEFAULT_BLOCK_INDICES: Sequence[int] = (8, 27, 46)


def _lazy_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "matplotlib is required for heatmap generation. "
            "Install it with `pip install matplotlib` in your environment."
        ) from exc
    return plt


def _downsample_matrix(weight: torch.Tensor, max_dim: int = 1024) -> Tuple[torch.Tensor, bool]:
    """
    Downsample a 2D tensor if either dimension exceeds max_dim to keep heatmaps tractable.

    Returns:
        (possibly downsampled tensor, did_downsample flag)
    """
    h, w = weight.shape
    if h <= max_dim and w <= max_dim:
        return weight, False

    new_h = min(h, max_dim)
    new_w = min(w, max_dim)
    weight_4d = weight.unsqueeze(0).unsqueeze(0).float()
    weight_ds = F.interpolate(weight_4d, size=(new_h, new_w), mode="area").squeeze(0).squeeze(0)
    return weight_ds, True


def _extract_weight_tensors(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """
    Return a list of (suffix, 2D weight tensor) from the provided module.

    Special handling:
    - MLP: return separate (fc1, fc2) weights.
    - ModulateDiT: use the linear projection weight.
    """
    tensors: List[Tuple[str, torch.Tensor]] = []

    if isinstance(module, MLP):
        tensors.append(("fc1", module.fc1.weight.detach()))
        tensors.append(("fc2", module.fc2.weight.detach()))
    elif isinstance(module, ModulateDiT):
        tensors.append(("", module.linear.weight.detach()))
    else:
        weight = getattr(module, "weight", None)
        if weight is None:
            return tensors
        tensors.append(("", weight.detach()))

    cleaned: List[Tuple[str, torch.Tensor]] = []
    for suffix, weight in tensors:
        if weight.ndim > 2:
            weight = weight.flatten(1)
        elif weight.ndim == 1:
            weight = weight.unsqueeze(0)
        cleaned.append((suffix, weight))

    return cleaned


def _save_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    plt = _lazy_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Clip extreme outliers and enhance contrast
    vmin, vmax = np.percentile(matrix, [1, 99])
    if vmin == vmax:
        vmin, vmax = matrix.min(), matrix.max()
    if vmin == vmax:
        vmin, vmax = -1e-6, 1e-6
    cmap = plt.get_cmap("RdBu_r").with_extremes(under="#1f2040", over="#ffb000")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Input dimension")
    ax.set_ylabel("Output dimension / stacked weights")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_worldplay_heatmaps(
    transformer,
    output_dir: Path | str = "outputs/heatmaps",
    block_indices: Iterable[int] = DEFAULT_BLOCK_INDICES,
    layers: Sequence[str] = DEFAULT_LAYER_ORDER,
) -> List[Path]:
    """
    Generate weight heatmaps for selected double-stream blocks.

    Args:
        transformer: A HunyuanVideo_1_5_DiffusionTransformer instance with double_blocks.
        output_dir: Directory to write PNG heatmaps into.
        block_indices: 1-based indices of double-stream blocks to visualize.
        layers: Layer attribute names to visualize inside each block.

    Returns:
        List of paths to the saved heatmap images.
    """
    if not hasattr(transformer, "double_blocks"):
        raise ValueError("Transformer is missing double_blocks; expected a worldplay transformer instance.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    num_blocks = len(transformer.double_blocks)

    for block_idx in block_indices:
        zero_idx = block_idx - 1
        if zero_idx < 0 or zero_idx >= num_blocks:
            logger.warning(
                "Requested block %s but transformer has %s double-stream blocks; skipping.",
                block_idx,
                num_blocks,
            )
            continue

        block = transformer.double_blocks[zero_idx]

        for layer_name in layers:
            layer = getattr(block, layer_name, None)
            if layer is None:
                logger.warning(
                    "Block %s is missing layer %s; skipping.", block_idx, layer_name
                )
                continue

            tensors = _extract_weight_tensors(layer)
            if not tensors:
                logger.warning(
                    "Layer %s on block %s has no weight tensor; skipping.", layer_name, block_idx
                )
                continue

            for suffix, weight in tensors:
                weight, is_downsampled = _downsample_matrix(weight)
                matrix = weight.float().cpu().numpy()
                title = f"Block {block_idx} · {layer_name}"
                if suffix:
                    title = f"{title} · {suffix}"
                combine_note = LAYER_COMBINE_NOTES.get(layer_name)
                if combine_note:
                    title = f"{title} ({combine_note})"
                if is_downsampled:
                    title = f"{title} (downsampled)"

                filename_suffix = f"_{suffix}" if suffix else ""
                filename = output_root / f"block{block_idx:02d}_{layer_name}{filename_suffix}.png"
                _save_heatmap(matrix, title, filename)
                saved_paths.append(filename)

    return saved_paths


__all__ = [
    "DEFAULT_BLOCK_INDICES",
    "DEFAULT_LAYER_ORDER",
    "generate_worldplay_heatmaps",
    "LAYER_COMBINE_NOTES",
]
