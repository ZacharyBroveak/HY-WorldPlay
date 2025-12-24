#!/usr/bin/env python3
"""
Print a layer-wise summary of the HY-WorldPlay transformer.

This script mirrors the inspection workflow we used for MatrixGame:
- builds the HunyuanVideo-1.5 transformer (optionally from a downloaded config)
- keeps parameters on the meta device by default to avoid large allocations
- walks the module tree and reports per-layer parameter counts

Usage examples:
  python print_architecture.py --max-depth 3
  python print_architecture.py --model-root /path/to/HunyuanVideo-1.5 --transformer-version 480p_i2v
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise printout of HY-WorldPlay")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Optional path to a downloaded HunyuanVideo-1.5 checkout (expects transformer/<version>/config.json). "
        "If omitted, the config defaults from the source code are used.",
    )
    parser.add_argument(
        "--transformer-version",
        type=str,
        default="480p_i2v",
        help="Transformer subfolder to read the config from (default: 480p_i2v).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Depth of module traversal. Use -1 to print every nested layer.",
    )
    parser.add_argument(
        "--safetensors-path",
        type=Path,
        default=None,
        help="If provided, summarize this safetensors checkpoint header instead of instantiating the model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model_arch.txt"),
        help="File path to save the summary alongside printing to stdout.",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Instantiate the model on CPU instead of the meta device (uses a lot more memory).",
    )
    parser.add_argument(
        "--skip-action",
        action="store_true",
        help="Do not attach the discrete-action conditioning layers.",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default="12355",
        help="Port used to bypass distributed init (defaults to 12355).",
    )
    return parser.parse_args()


def bootstrap_env(master_port: str) -> None:
    """Pre-set distributed env vars so hyvideo import does not try to open sockets."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", master_port)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def count_local_params(module: torch.nn.Module) -> tuple[int, int]:
    """Return (total) parameter counts for the module only (no children)."""
    params = list(module.parameters(recurse=False))
    total = sum(p.numel() for p in params)
    return total


def read_safetensors_header(path: Path) -> Dict[str, dict]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header


def build_param_tree(header: Dict[str, dict]) -> Dict:
    """
    Build a nested tree from safetensors header keys to aggregate parameter counts.
    """
    root = {"__params__": 0, "children": {}, "__shape__": None}
    for key, meta in header.items():
        shape = meta["shape"]
        numel = 1
        for dim in shape:
            numel *= dim
        parts = key.split(".")
        node = root
        for part in parts:
            node = node["children"].setdefault(part, {"__params__": 0, "children": {}, "__shape__": None})
        node["__params__"] += numel
        node["__shape__"] = shape
    return root


def compute_totals(node: Dict) -> int:
    total_local = node.get("__params__", 0)
    total_children = sum(compute_totals(child) for child in node["children"].values())
    node["__total__"] = total_local + total_children
    return node["__total__"]


def walk_tree(node: Dict, name: str, depth: int, max_depth: Optional[int], lines: List[str]) -> None:
    indent = "  " * depth
    total = node.get("__total__", 0)
    local = node.get("__params__", 0)
    shape = node.get("__shape__")
    shape_str = f" shape={tuple(shape)}" if shape else ""
    lines.append(f"{indent}{name}: params(total)={total:,} local={local:,}{shape_str}")
    if max_depth is not None and max_depth >= 0 and depth >= max_depth:
        return
    def _sort_key(item):
        k = item[0]
        return (0, int(k)) if k.isdigit() else (1, k)

    for child_name, child in sorted(node["children"].items(), key=_sort_key):
        walk_tree(child, child_name, depth + 1, max_depth, lines)


def walk_module(
    module: torch.nn.Module,
    name: str,
    depth: int,
    max_depth: Optional[int],
    lines: List[str],
) -> None:
    """Recursively walk modules and append formatted lines to `lines`."""
    total = count_local_params(module)
    indent = "  " * depth
    lines.append(f"{indent}{name}: {module.__class__.__name__} | params={total:,}")

    if max_depth is not None and max_depth >= 0 and depth >= max_depth:
        return

    for child_name, child in module.named_children():
        walk_module(child, child_name, depth + 1, max_depth, lines)


def build_model(args: argparse.Namespace):
    """Instantiate the transformer, optionally using a downloaded config."""
    if not args.no_meta:
        torch.set_default_device("meta")

    # Import after env/bootstrap to avoid side effects during module import.
    from hyvideo.models.transformers.worldplay_1_5_transformer import (
        HunyuanVideo_1_5_DiffusionTransformer,
    )

    if args.model_root is not None:
        transformer_dir = args.model_root / "transformer" / args.transformer_version
        if not transformer_dir.exists():
            raise FileNotFoundError(
                f"Could not find transformer config at: {transformer_dir} "
                "(pass a valid --model-root or omit it to use code defaults)."
            )
        model = HunyuanVideo_1_5_DiffusionTransformer.from_config(transformer_dir)
    else:
        model = HunyuanVideo_1_5_DiffusionTransformer()

    if not args.skip_action:
        model.add_action_parameters()

    return model


def main():
    args = parse_args()
    bootstrap_env(args.master_port)

    max_depth = None if args.max_depth < 0 else args.max_depth
    lines: List[str] = []

    if args.safetensors_path is not None:
        header = read_safetensors_header(args.safetensors_path)
        tree = build_param_tree(header)
        compute_totals(tree)
        lines.extend(
            [
                f"HY-WorldPlay checkpoint summary from {args.safetensors_path}",
                "Derived from safetensors header (no model instantiation).",
                "",
            ]
        )
        walk_tree(tree, "checkpoint", depth=0, max_depth=max_depth, lines=lines)
    else:
        model = build_model(args)

        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        lines.extend(
            [
                f"HY-WorldPlay transformer summary (trainable/total params={total_trainable:,}/{total_params:,})",
                f"Config source: {'defaults from code' if args.model_root is None else args.model_root}",
                f"Action conditioning: {'enabled' if not args.skip_action else 'disabled'}",
                f"Meta device: {'yes (no large allocations)' if not args.no_meta else 'no (real tensors)'}",
                "",
            ]
        )

        walk_module(model, "transformer", depth=0, max_depth=max_depth, lines=lines)

    summary_text = "\n".join(lines)
    print(summary_text)

    if args.output is not None:
        if not args.output.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {args.output.parent}. "
                "Create it manually or pass a different --output path."
            )
        args.output.write_text(summary_text)
        print(f"\nSaved summary to: {args.output}")


if __name__ == "__main__":
    main()
