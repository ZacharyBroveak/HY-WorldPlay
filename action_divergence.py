#!/usr/bin/env python3
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import argparse
import json
from PIL import Image
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

ACTION_SPACE = 81

ACTION_REGIME_MAP = {
    "noop": 0,
    "forward": 9,
    "backward": 18,
    "right": 27,
    "left": 36,
    "yaw_right": 1,
    "yaw_left": 2,
    "pitch_up": 3,
    "pitch_down": 4,
}

TRANS_SHORT = [
    "S",
    "F",
    "B",
    "R",
    "L",
    "F+R",
    "F+L",
    "B+R",
    "B+L",
]
ROT_SHORT = [
    "N",
    "R",
    "L",
    "U",
    "D",
    "R+U",
    "R+D",
    "L+U",
    "L+D",
]


def _lazy_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "matplotlib is required for plotting divergence heatmaps. "
            "Install it with `pip install matplotlib`."
        ) from exc
    return plt


def parse_index_list(value: str, *, min_value: int, max_value: int) -> List[int]:
    if not value:
        return []
    items: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            items.extend(range(start, end + 1))
        else:
            items.append(int(part))
    unique = sorted(set(items))
    for idx in unique:
        if idx < min_value or idx > max_value:
            raise ValueError(f"Index {idx} is out of bounds [{min_value}, {max_value}].")
    return unique


def parse_actions(value: Optional[str]) -> List[int]:
    if value is None:
        return []
    if value.strip().lower() == "all":
        return list(range(ACTION_SPACE))
    return parse_index_list(value, min_value=0, max_value=ACTION_SPACE - 1)


def parse_action_regimes(value: Optional[str]) -> List[int]:
    if value is None:
        return []
    regimes: List[int] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part.isdigit():
            regimes.append(int(part))
            continue
        if part not in ACTION_REGIME_MAP:
            raise ValueError(f"Unknown action regime '{part}'. Known: {sorted(ACTION_REGIME_MAP.keys())}")
        regimes.append(ACTION_REGIME_MAP[part])
    for action_id in regimes:
        if action_id < 0 or action_id >= ACTION_SPACE:
            raise ValueError(f"Action id {action_id} is out of bounds [0, {ACTION_SPACE - 1}].")
    return regimes


def parse_action_id(value: str) -> int:
    value = value.strip().lower()
    if value.isdigit():
        action_id = int(value)
    else:
        if value not in ACTION_REGIME_MAP:
            raise ValueError(f"Unknown base action '{value}'. Known: {sorted(ACTION_REGIME_MAP.keys())}")
        action_id = ACTION_REGIME_MAP[value]
    if action_id < 0 or action_id >= ACTION_SPACE:
        raise ValueError(f"Action id {action_id} is out of bounds [0, {ACTION_SPACE - 1}].")
    return action_id


def describe_action(action_id: int) -> str:
    trans = action_id // 9
    rot = action_id % 9
    return f"{TRANS_SHORT[trans]}/{ROT_SHORT[rot]}"


def prepare_camera_inputs(device: torch.device, frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
    viewmats = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(1, frames, 1, 1)
    Ks = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(1, frames, 1, 1)
    return viewmats, Ks


def init_kv_cache(transformer, *, device: torch.device, dtype: torch.dtype) -> List[Dict[str, Optional[torch.Tensor]]]:
    head_dim = transformer.hidden_size // transformer.heads_num
    kv_cache: List[Dict[str, Optional[torch.Tensor]]] = []
    for _ in transformer.double_blocks:
        empty_txt = torch.zeros((1, transformer.heads_num, 0, head_dim), device=device, dtype=dtype)
        kv_cache.append({
            "k_vision": None,
            "v_vision": None,
            "q_vision": None,
            "k_txt": empty_txt,
            "v_txt": empty_txt,
        })
    return kv_cache


def prepare_latents(
    pipe: HunyuanVideo_1_5_Pipeline,
    *,
    device: torch.device,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    seed: int,
    task_type: str,
    image_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.in_channels,
        latent_height=latent_height,
        latent_width=latent_width,
        video_length=latent_frames,
        dtype=transformer_dtype,
        device=device,
        generator=generator,
    )

    image_latents = None

    if image_path and task_type == "i2v":
        image = Image.open(image_path).convert("RGB")

        # Replicate image processing from the main pipeline.
        # The VAE scale factor is used to determine the pixel dimensions from latent dimensions.
        height = latent_height * pipe.vae_scale_factor
        width = latent_width * pipe.vae_scale_factor

        processed_image = pipe.image_processor.preprocess(image, height=height, width=width).to(
            device=device, dtype=transformer_dtype
        )

        # Encode the image into the latent space using the VAE.
        # Use the same generator for reproducibility.
        image_latents = pipe.vae.encode(processed_image.unsqueeze(2)).latent_dist.sample(generator=generator)
        image_latents = image_latents * pipe.vae.config.scaling_factor

    # Manually prepare conditioning latents to ensure image is used correctly.
    # The pipeline's _prepare_cond_latents method appears to incorrectly discard
    # the image conditioning in this manual execution context.
    cond_latents = torch.zeros_like(latents)
    if task_type == "i2v" and image_latents is not None:
        cond_latents[:, :, :1] = image_latents

    task_mask = pipe.get_task_mask(task_type, latent_frames)

    return latents, cond_latents, task_mask


def run_latent_sequence(
    pipe: HunyuanVideo_1_5_Pipeline,
    *,
    latents_init: torch.Tensor,
    cond_latents_init: torch.Tensor,
    task_mask: torch.Tensor,
    timesteps: torch.Tensor,
    action_sequence: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    task_type: str,
) -> List[torch.Tensor]:
    latents = latents_init.clone()
    cond_latents = cond_latents_init.clone()
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    autocast_enabled = latents.device.type == "cuda"
    viewmats = viewmats.to(device=latents.device, dtype=transformer_dtype)
    Ks = Ks.to(device=latents.device, dtype=transformer_dtype)
    action_input = action_sequence.to(device=latents.device, dtype=transformer_dtype).unsqueeze(0)
    kv_cache = init_kv_cache(pipe.transformer, device=latents.device, dtype=latents.dtype)
    latent_steps: List[torch.Tensor] = []

    mask = task_mask.to(device=latents.device, dtype=latents.dtype).view(1, 1, -1, 1, 1).expand(
        latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]
    )

    with torch.inference_mode():
        for t in timesteps:
            timestep_input = torch.full((latents.shape[2],), t, device=latents.device, dtype=timesteps.dtype)
            latents_concat = torch.concat([latents, cond_latents, mask], dim=1)
            latents_concat = pipe.scheduler.scale_model_input(latents_concat, t)
            latents_concat = latents_concat.to(dtype=transformer_dtype)
            with torch.autocast(device_type=latents.device.type, dtype=transformer_dtype, enabled=autocast_enabled):
                noise_pred = pipe.transformer.forward_vision(
                    hidden_states=latents_concat,
                    timestep=timestep_input,
                    action=action_input,
                    viewmats=viewmats,
                    Ks=Ks,
                    mask_type=task_type,
                    kv_cache=kv_cache,
                    cache_vision=False,
                    rope_temporal_size=latents_concat.shape[2],
                    start_rope_start_idx=0,
                )[0]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latent_steps.append(latents.detach())

    return latent_steps


def save_curve(path: Path, curve: Sequence[float], meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), np.array(curve, dtype=np.float32))
    payload = {"divergence": list(map(float, curve))}
    payload.update(meta)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2))


def save_maps(path: Path, maps: np.ndarray, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), maps.astype(np.float32))
    payload = {
        "maps_path": str(path.with_suffix(".npy")),
        "shape": list(maps.shape),
    }
    payload.update(meta)
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2))


def plot_heatmap(matrix: np.ndarray, title: str, path: Path, *, vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    plt = _lazy_matplotlib()
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("Step")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def has_quantized_mod_layer(transformer, quantized_linear_cls: type) -> bool:
    for block in getattr(transformer, "double_blocks", []):
        for mod_name in ("img_mod", "txt_mod"):
            mod = getattr(block, mod_name, None)
            linear = getattr(mod, "linear", None) if mod is not None else None
            if isinstance(linear, quantized_linear_cls):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute action-vs-base latent divergence per step.")
    parser.add_argument("--model-path", required=True, help="Path to the HY-WorldPlay model directory.")
    parser.add_argument("--transformer-version", default="480p_i2v", help="Transformer version folder name.")
    parser.add_argument("--action-ckpt", required=True, help="Path to the action checkpoint safetensors.")
    parser.add_argument("--actions", default=None, help="Action ids list (e.g. 0,9,27) or 'all'.")
    parser.add_argument("--action-regimes", default=None, help="Named regimes: noop,forward,right,yaw_right,...")
    parser.add_argument("--base-action", required=True, help="Base action id or name (e.g. noop, 0).")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--output-dir", default="outputs/action_divergence", help="Directory to write outputs.")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"), help="Transformer dtype.")
    parser.add_argument("--device", default="cuda", help="Device for inference (cuda or cpu).")
    parser.add_argument("--latent-frames", type=int, default=4, help="Latent frame count.")
    parser.add_argument("--latent-height", type=int, default=32, help="Latent height in VAE space.")
    parser.add_argument("--latent-width", type=int, default=32, help="Latent width in VAE space.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for latents.")
    parser.add_argument("--task-type", default="t2v", choices=("t2v", "i2v"), help="Task type for masking.")
    parser.add_argument("--image-path", default=None, help="Path to input image for i2v task.")
    parser.add_argument("--save-maps", action="store_true", help="Save per-step divergence maps.")
    parser.add_argument("--save-channel-maps", action="store_true", help="Save per-step divergence maps per channel.")
    parser.add_argument("--plot-heatmap", action="store_true", help="Plot step x token heatmap.")
    parser.add_argument("--plot-channel-heatmaps", action="store_true", help="Plot per-channel step x token heatmaps.")
    parser.add_argument("--heatmap-vmin", type=float, default=None, help="Fixed heatmap min value.")
    parser.add_argument("--heatmap-vmax", type=float, default=None, help="Fixed heatmap max value.")
    parser.add_argument("--fixed-heatmap-scale", action="store_true", help="Use a fixed color scale across actions.")
    parser.add_argument("--smoothquant", action="store_true", help="Enable SmoothQuant-style fake quantization.")
    parser.add_argument("--sq-double-bits", type=int, default=8, help="SmoothQuant bitwidth for double blocks.")
    parser.add_argument("--sq-single-bits", type=int, default=8, help="SmoothQuant bitwidth for single blocks.")
    parser.add_argument("--sq-final-bits", type=int, default=None, help="SmoothQuant bitwidth for final layer.")
    parser.add_argument("--sq-cond-bits", type=int, default=None, help="SmoothQuant bitwidth for conditioning layers.")
    parser.add_argument("--sq-layer-bits", type=str, default=None, help="JSON dict of layer bitwidth overrides.")
    parser.add_argument("--quantize-actions-only", action="store_true", help="Quantize after base run so no-op stays unquantized.")
    parser.add_argument("--enable-torch-compile", action="store_true", help="Enable torch.compile if supported.")
    args = parser.parse_args()

    initialize_infer_state(args)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    output_dir = Path(args.output_dir)

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=args.transformer_version,
        enable_offloading=False,
        enable_group_offloading=False,
        transformer_dtype=dtype,
        action_ckpt=args.action_ckpt,
        device=device,
    )
    pipe.vae.to(dtype=dtype)
    pipe.transformer.eval()

    layer_bits = None
    if args.sq_layer_bits:
        try:
            layer_bits = json.loads(args.sq_layer_bits)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for --sq-layer-bits: {args.sq_layer_bits}") from exc

    def apply_smoothquant() -> bool:
        from smoothquant.fake_quant import W8A8Linear, quantize_hyworldplay

        quantize_hyworldplay(
            pipe.transformer,
            double_n_bits=args.sq_double_bits,
            single_n_bits=args.sq_single_bits,
            final_n_bits=args.sq_final_bits,
            cond_n_bits=args.sq_cond_bits,
            layer_bits=layer_bits,
        )
        return has_quantized_mod_layer(pipe.transformer, W8A8Linear)

    mod_quantized = False
    if args.smoothquant and not args.quantize_actions_only:
        mod_quantized = apply_smoothquant()

    base_action_id = parse_action_id(args.base_action)
    action_ids = parse_actions(args.actions)
    action_ids.extend(parse_action_regimes(args.action_regimes))
    if not action_ids:
        raise ValueError("Provide --actions or --action-regimes.")
    action_ids = [action_id for action_id in sorted(set(action_ids)) if action_id != base_action_id]

    viewmats, Ks = prepare_camera_inputs(device, args.latent_frames)

    latents_init, cond_latents_init, task_mask = prepare_latents(
        pipe,
        device=device,
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        seed=args.seed,
        task_type=args.task_type,
        image_path=args.image_path,
    )

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    base_sequence = torch.full((args.latent_frames,), base_action_id, device=device, dtype=torch.long)
    base_latent_steps = run_latent_sequence(
        pipe,
        latents_init=latents_init,
        cond_latents_init=cond_latents_init,
        task_mask=task_mask,
        timesteps=timesteps,
        action_sequence=base_sequence,
        viewmats=viewmats,
        Ks=Ks,
        task_type=args.task_type,
    )

    if args.smoothquant and args.quantize_actions_only:
        mod_quantized = apply_smoothquant()

    heatmap_items: List[Tuple[np.ndarray, str, Path]] = []
    channel_heatmap_items: List[Tuple[np.ndarray, str, Path]] = []
    heatmap_min = None
    heatmap_max = None
    channel_min = None
    channel_max = None

    filename_prefix = "quantized_mod_" if mod_quantized else ""

    for action_id in action_ids:
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        action_sequence = torch.full((args.latent_frames,), action_id, device=device, dtype=torch.long)
        latent_steps = run_latent_sequence(
            pipe,
            latents_init=latents_init,
            cond_latents_init=cond_latents_init,
            task_mask=task_mask,
            timesteps=timesteps,
            action_sequence=action_sequence,
            viewmats=viewmats,
            Ks=Ks,
            task_type=args.task_type,
        )

        divergence_curve: List[float] = []
        divergence_maps: List[np.ndarray] = []
        channel_maps: Optional[List[List[np.ndarray]]] = None
        for step_idx, latents in enumerate(latent_steps):
            base_latents = base_latent_steps[step_idx]
            diff = (latents - base_latents).float()
            divergence_curve.append(torch.linalg.norm(diff).item())
            per_token = diff.pow(2).sum(dim=1).sqrt().squeeze(0).cpu().numpy()
            divergence_maps.append(per_token)
            if args.save_channel_maps or args.plot_channel_heatmaps:
                per_channel = diff.abs().squeeze(0).cpu().numpy()
                if channel_maps is None:
                    channel_maps = [[] for _ in range(per_channel.shape[0])]
                for channel_idx in range(per_channel.shape[0]):
                    channel_maps[channel_idx].append(per_channel[channel_idx])

        action_label = describe_action(action_id).replace("/", "_").replace("+", "plus")
        meta = {
            "action_id": action_id,
            "action_label": describe_action(action_id),
            "base_action_id": base_action_id,
            "base_action_label": describe_action(base_action_id),
            "num_steps": len(timesteps),
        }
        curve_path = output_dir / f"{filename_prefix}divergence_curve_{action_label}"
        save_curve(curve_path, divergence_curve, meta)

        if args.save_maps or args.plot_heatmap:
            maps_array = np.stack(divergence_maps, axis=0)
            maps_path = output_dir / f"{filename_prefix}divergence_maps_{action_label}"
            save_maps(maps_path, maps_array, meta)

        if args.save_channel_maps and channel_maps is not None:
            for channel_idx, maps in enumerate(channel_maps):
                maps_array = np.stack(maps, axis=0)
                maps_path = output_dir / f"{filename_prefix}divergence_maps_{action_label}_ch{channel_idx}"
                channel_meta = dict(meta)
                channel_meta["channel_index"] = channel_idx
                save_maps(maps_path, maps_array, channel_meta)

        if args.plot_heatmap:
            heatmap = maps_array.reshape(maps_array.shape[0], -1)
            heatmap_items.append((
                heatmap,
                f"Divergence {describe_action(action_id)} vs {describe_action(base_action_id)}",
                output_dir / f"{filename_prefix}divergence_heatmap_{action_label}.png",
            ))
            if args.fixed_heatmap_scale and args.heatmap_vmin is None and args.heatmap_vmax is None:
                heatmap_min = float(heatmap.min()) if heatmap_min is None else min(heatmap_min, float(heatmap.min()))
                heatmap_max = float(heatmap.max()) if heatmap_max is None else max(heatmap_max, float(heatmap.max()))

        if args.plot_channel_heatmaps and channel_maps is not None:
            for channel_idx, maps in enumerate(channel_maps):
                maps_array = np.stack(maps, axis=0)
                heatmap = maps_array.reshape(maps_array.shape[0], -1)
                channel_heatmap_items.append((
                    heatmap,
                    (
                        f"Divergence {describe_action(action_id)} vs "
                        f"{describe_action(base_action_id)} (ch {channel_idx})"
                    ),
                    output_dir / f"{filename_prefix}divergence_heatmap_{action_label}_ch{channel_idx}.png",
                ))
                if args.fixed_heatmap_scale and args.heatmap_vmin is None and args.heatmap_vmax is None:
                    channel_min = float(heatmap.min()) if channel_min is None else min(channel_min, float(heatmap.min()))
                    channel_max = float(heatmap.max()) if channel_max is None else max(channel_max, float(heatmap.max()))

    if heatmap_items:
        heatmap_vmin = args.heatmap_vmin if args.heatmap_vmin is not None else (heatmap_min if args.fixed_heatmap_scale else None)
        heatmap_vmax = args.heatmap_vmax if args.heatmap_vmax is not None else (heatmap_max if args.fixed_heatmap_scale else None)
        for heatmap, title, path in heatmap_items:
            plot_heatmap(heatmap, title=title, path=path, vmin=heatmap_vmin, vmax=heatmap_vmax)

    if channel_heatmap_items:
        channel_vmin = args.heatmap_vmin if args.heatmap_vmin is not None else (channel_min if args.fixed_heatmap_scale else None)
        channel_vmax = args.heatmap_vmax if args.heatmap_vmax is not None else (channel_max if args.fixed_heatmap_scale else None)
        for heatmap, title, path in channel_heatmap_items:
            plot_heatmap(heatmap, title=title, path=path, vmin=channel_vmin, vmax=channel_vmax)

    print(f"Saved divergence outputs to {output_dir}")


if __name__ == "__main__":
    main()
