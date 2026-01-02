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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline

ACTION_SPACE = 81

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
TRANS_LONG = [
    "stay",
    "forward",
    "backward",
    "right",
    "left",
    "forward+right",
    "forward+left",
    "backward+right",
    "backward+left",
]
ROT_LONG = [
    "none",
    "yaw_right",
    "yaw_left",
    "pitch_up",
    "pitch_down",
    "yaw_right+pitch_up",
    "yaw_right+pitch_down",
    "yaw_left+pitch_up",
    "yaw_left+pitch_down",
]

ACTION_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}

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


def _lazy_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "matplotlib is required for plotting CKA curves. Install it with `pip install matplotlib`."
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
    regimes = []
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


def parse_layers(value: Optional[str], max_layers: int) -> List[int]:
    if value is None or value.strip().lower() == "all":
        return list(range(1, max_layers + 1))
    return parse_index_list(value, min_value=1, max_value=max_layers)


def describe_action(action_id: int) -> Tuple[str, str]:
    trans = action_id // 9
    rot = action_id % 9
    short = f"{TRANS_SHORT[trans]}/{ROT_SHORT[rot]}"
    long = f"{TRANS_LONG[trans]}|{ROT_LONG[rot]}"
    return short, long


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: {x.shape} vs {y.shape}")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    dot_xy = x.T @ y
    numerator = (dot_xy * dot_xy).sum()
    dot_xx = x.T @ x
    dot_yy = y.T @ y
    denom = torch.sqrt((dot_xx * dot_xx).sum() * (dot_yy * dot_yy).sum()) + eps
    return (numerator / denom).item()


def one_hot_to_one_dimension(one_hot: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [ACTION_MAPPING[tuple(row.tolist())] for row in one_hot],
        dtype=torch.long,
    )


def pose_to_input(pose_json_path: str, *, tps: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose_json = json.load(open(pose_json_path, "r"))
    pose_keys = list(pose_json.keys())
    intrinsic_list = []
    w2c_list = []
    for key in pose_keys:
        c2w = np.array(pose_json[key]["extrinsic"])
        w2c = np.linalg.inv(c2w)
        w2c_list.append(w2c)
        intrinsic = np.array(pose_json[key]["K"])
        intrinsic[0, 0] /= intrinsic[0, 2] * 2
        intrinsic[1, 1] /= intrinsic[1, 2] * 2
        intrinsic[0, 2] = 0.5
        intrinsic[1, 2] = 0.5
        intrinsic_list.append(intrinsic)

    w2c_list = np.array(w2c_list)
    intrinsic_list = torch.tensor(np.array(intrinsic_list))

    c2ws = np.linalg.inv(w2c_list)
    C_inv = np.linalg.inv(c2ws[:-1])
    relative_c2w = np.zeros_like(c2ws)
    relative_c2w[0, ...] = c2ws[0, ...]
    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

    move_norm_valid = 0.0001
    for i in range(1, relative_c2w.shape[0]):
        move_dirs = relative_c2w[i, :3, 3]
        move_norms = np.linalg.norm(move_dirs)
        if move_norms > move_norm_valid:
            move_norm_dirs = move_dirs / move_norms
            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
            trans_angles_deg = angles_rad * (180.0 / np.pi)
        else:
            trans_angles_deg = np.zeros(3)

        R_rel = relative_c2w[i, :3, :3]
        r = R.from_matrix(R_rel)
        rot_angles_deg = r.as_euler("xyz", degrees=True)

        if move_norms > move_norm_valid:
            if (not tps) or (tps and abs(rot_angles_deg[1]) < 5e-2 and abs(rot_angles_deg[0]) < 5e-2):
                if trans_angles_deg[2] < 60:
                    trans_one_hot[i, 0] = 1
                elif trans_angles_deg[2] > 120:
                    trans_one_hot[i, 1] = 1

                if trans_angles_deg[0] < 60:
                    trans_one_hot[i, 2] = 1
                elif trans_angles_deg[0] > 120:
                    trans_one_hot[i, 3] = 1

        if rot_angles_deg[1] > 5e-2:
            rotate_one_hot[i, 0] = 1
        elif rot_angles_deg[1] < -5e-2:
            rotate_one_hot[i, 1] = 1

        if rot_angles_deg[0] > 5e-2:
            rotate_one_hot[i, 2] = 1
        elif rot_angles_deg[0] < -5e-2:
            rotate_one_hot[i, 3] = 1

    trans_one_hot = torch.tensor(trans_one_hot)
    rotate_one_hot = torch.tensor(rotate_one_hot)

    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
    action_one_label = trans_one_label * 9 + rotate_one_label

    return torch.tensor(w2c_list), intrinsic_list, action_one_label


@dataclass
class SampleIndexCache:
    indices: Optional[torch.Tensor] = None

    def select(self, total: int, *, max_samples: Optional[int], seed: int, device: torch.device) -> Optional[torch.Tensor]:
        if max_samples is None or total <= max_samples:
            return None
        if self.indices is None or self.indices.numel() != max_samples:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            self.indices = torch.randperm(total, generator=generator, device=device)[:max_samples].cpu()
        return self.indices.to(device)


def prepare_latents(
    pipe: HunyuanVideo_1_5_Pipeline,
    *,
    device: torch.device,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    seed: int,
    task_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    task_mask = pipe.get_task_mask(task_type, latent_frames)
    cond_latents = pipe._prepare_cond_latents(task_type, None, latents, task_mask)
    return latents, cond_latents


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


def flatten_noise_pred(noise_pred: torch.Tensor, indices: Optional[torch.Tensor]) -> torch.Tensor:
    samples = noise_pred.permute(0, 2, 3, 4, 1).reshape(-1, noise_pred.shape[1])
    if indices is not None:
        samples = samples.index_select(0, indices)
    return samples.detach().float().cpu()


class BlockActivationCollector:
    def __init__(self, token_sampler: SampleIndexCache, max_tokens: Optional[int], seed: int):
        self.token_sampler = token_sampler
        self.max_tokens = max_tokens
        self.seed = seed
        self.activations: Dict[int, torch.Tensor] = {}

    def add(self, layer_idx: int, activation: torch.Tensor) -> None:
        token_count = activation.shape[1]
        indices = self.token_sampler.select(token_count, max_samples=self.max_tokens, seed=self.seed, device=activation.device)
        if indices is not None:
            activation = activation.index_select(1, indices)
        activation = activation.reshape(-1, activation.shape[-1]).detach().float().cpu()
        self.activations[layer_idx] = activation


def build_block_hook(collector: BlockActivationCollector, layer_idx: int):
    def hook(_module, _inputs, output):
        activation = output[0] if isinstance(output, (tuple, list)) else output
        collector.add(layer_idx, activation)
    return hook


def save_curve(output_dir: Path, name: str, curve: Sequence[float], meta: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{name}.npy", np.array(curve, dtype=np.float32))
    payload = {"cka": list(map(float, curve))}
    payload.update(meta)
    (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))


def save_block_curve(
    output_dir: Path,
    name: str,
    matrix: np.ndarray,
    layer_ids: Sequence[int],
    meta: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{name}.npy", matrix.astype(np.float32))
    payload = {"layers": list(layer_ids), "cka": matrix.tolist()}
    payload.update(meta)
    (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))


def plot_curve(curve: Sequence[float], title: str, path: Path) -> None:
    plt = _lazy_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(range(1, len(curve) + 1), curve, marker="o", linewidth=1.5)
    ax.set_xlabel("Step t -> t+1")
    ax.set_ylabel("CKA")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute step-to-step CKA curves per action regime.")
    parser.add_argument("--model-path", required=True, help="Path to the HY-WorldPlay model directory.")
    parser.add_argument("--transformer-version", default="480p_i2v", help="Transformer version folder name.")
    parser.add_argument("--action-ckpt", required=True, help="Path to the action checkpoint safetensors.")
    parser.add_argument("--actions", default=None, help="Action ids list (e.g. 0,9,27) or 'all'.")
    parser.add_argument("--action-regimes", default=None, help="Named regimes: noop,forward,right,yaw_right,...")
    parser.add_argument("--pose-json", default=None, help="Optional pose JSON for viewmats/Ks/action sequence.")
    parser.add_argument("--tps", action="store_true", help="Enable TPS motion gate in pose action extraction.")
    parser.add_argument("--layers", default=None, help="1-based layer indices (e.g. 1,5,10) or 'all'.")
    parser.add_argument("--output-dir", default="outputs/cka_steps", help="Directory to write outputs.")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"), help="Transformer dtype.")
    parser.add_argument("--device", default="cuda", help="Device for inference (cuda or cpu).")
    parser.add_argument("--latent-frames", type=int, default=None, help="Latent frame count.")
    parser.add_argument("--latent-height", type=int, default=32, help="Latent height in VAE space.")
    parser.add_argument("--latent-width", type=int, default=32, help="Latent width in VAE space.")
    parser.add_argument("--video-length", type=int, default=None, help="Optional video length in frames.")
    parser.add_argument("--height", type=int, default=None, help="Optional video height in pixels.")
    parser.add_argument("--width", type=int, default=None, help="Optional video width in pixels.")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for latents and sampling.")
    parser.add_argument("--max-noise-samples", type=int, default=16384, help="Max spatial samples for noise CKA.")
    parser.add_argument("--max-block-tokens", type=int, default=512, help="Max tokens per block for block CKA.")
    parser.add_argument("--task-type", default="t2v", choices=("t2v", "i2v"), help="Task type for masking.")
    parser.add_argument("--skip-blocks", action="store_true", help="Skip per-block CKA (only step CKA).")
    parser.add_argument("--enable-torch-compile", action="store_true", help="Enable torch.compile if supported.")
    args = parser.parse_args()

    initialize_infer_state(args)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=args.transformer_version,
        enable_offloading=False,
        enable_group_offloading=False,
        transformer_dtype=dtype,
        action_ckpt=args.action_ckpt,
        device=device,
    )
    transformer = pipe.transformer
    transformer.eval()

    layers = parse_layers(args.layers, max_layers=len(transformer.double_blocks))
    output_dir = Path(args.output_dir)

    pose_action_sequence = None
    pose_viewmats = None
    pose_Ks = None
    if args.pose_json:
        pose_viewmats, pose_Ks, pose_action_sequence = pose_to_input(args.pose_json, tps=args.tps)
        pose_frames = pose_action_sequence.shape[0]
        if args.video_length and args.height and args.width:
            latent_frames, latent_height, latent_width = pipe.get_latent_size(
                args.video_length, args.height, args.width
            )
            if latent_frames != pose_frames:
                raise ValueError(
                    f"Pose JSON has {pose_frames} latent frames, but video length implies {latent_frames}."
                )
        else:
            latent_frames = args.latent_frames or pose_frames
            if latent_frames != pose_frames:
                raise ValueError(
                    f"Pose JSON has {pose_frames} latent frames, but --latent-frames is {latent_frames}."
                )
            latent_height = args.latent_height
            latent_width = args.latent_width
    else:
        if args.video_length and args.height and args.width:
            latent_frames, latent_height, latent_width = pipe.get_latent_size(
                args.video_length, args.height, args.width
            )
        else:
            latent_frames = args.latent_frames or 4
            latent_height = args.latent_height
            latent_width = args.latent_width

    patch_size = transformer.config.patch_size
    if latent_frames % patch_size[0] != 0 or latent_height % patch_size[1] != 0 or latent_width % patch_size[2] != 0:
        raise ValueError(
            "Latent dimensions must be divisible by patch size "
            f"{patch_size} (got T={latent_frames}, H={latent_height}, W={latent_width})."
        )

    base_latents, base_cond_latents = prepare_latents(
        pipe,
        device=device,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        seed=args.seed,
        task_type=args.task_type,
    )

    if args.pose_json:
        viewmats = pose_viewmats.unsqueeze(0).to(device)
        Ks = pose_Ks.unsqueeze(0).to(device)
        action_sequences = {"pose": pose_action_sequence.to(device)}
    else:
        viewmats, Ks = prepare_camera_inputs(device, latent_frames)
        action_ids = parse_actions(args.actions)
        if args.action_regimes:
            action_ids.extend(parse_action_regimes(args.action_regimes))
        if not action_ids:
            raise ValueError("Provide --actions or --action-regimes when --pose-json is not set.")
        action_sequences = {str(action_id): torch.full((latent_frames,), action_id, device=device, dtype=torch.long)
                            for action_id in action_ids}

    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    noise_sampler = SampleIndexCache()
    block_sampler = SampleIndexCache()

    for name, action_sequence in action_sequences.items():
        pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        if name == "pose":
            action_label = "pose"
            action_id = None
        else:
            action_id = int(name)
            action_label, _ = describe_action(action_id)

        latents = base_latents.clone()
        cond_latents = base_cond_latents.clone()
        kv_cache = init_kv_cache(transformer, device=device, dtype=latents.dtype)

        step_curve: List[float] = []
        block_curve: Dict[int, List[float]] = {layer_idx: [] for layer_idx in layers}

        prev_noise = None
        prev_block_acts: Dict[int, torch.Tensor] = {}

        handles = []
        collector = None
        max_noise_samples = None if args.max_noise_samples <= 0 else args.max_noise_samples
        max_block_tokens = None if args.max_block_tokens <= 0 else args.max_block_tokens
        if not args.skip_blocks:
            collector = BlockActivationCollector(block_sampler, max_block_tokens, args.seed)
            for idx, block in enumerate(transformer.double_blocks, start=1):
                if idx in layers:
                    handles.append(block.register_forward_hook(build_block_hook(collector, idx)))

        try:
            for t in timesteps:
                timestep_input = torch.full((latent_frames,), t, device=device, dtype=timesteps.dtype)
                latents_concat = torch.concat([latents, cond_latents], dim=1)
                latents_concat = pipe.scheduler.scale_model_input(latents_concat, t)

                if autocast_enabled:
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype, enabled=True)
                else:
                    autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

                with torch.no_grad():
                    with autocast_ctx:
                        noise_pred = transformer.forward_vision(
                            hidden_states=latents_concat,
                            timestep=timestep_input,
                            action=action_sequence.unsqueeze(0),
                            viewmats=viewmats,
                            Ks=Ks,
                            mask_type=args.task_type,
                            kv_cache=kv_cache,
                            cache_vision=False,
                            rope_temporal_size=latents_concat.shape[2],
                            start_rope_start_idx=0,
                        )[0]

                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if prev_noise is not None:
                    total_samples = noise_pred.shape[0] * noise_pred.shape[2] * noise_pred.shape[3] * noise_pred.shape[4]
                    noise_indices = noise_sampler.select(
                        total_samples,
                        max_samples=max_noise_samples,
                        seed=args.seed,
                        device=noise_pred.device,
                    )
                    current_noise = flatten_noise_pred(noise_pred, noise_indices)
                    prev_noise_flat = flatten_noise_pred(prev_noise, noise_indices)
                    step_curve.append(linear_cka(prev_noise_flat, current_noise))

                    if collector is not None:
                        for layer_idx in layers:
                            current_act = collector.activations[layer_idx]
                            prev_act = prev_block_acts[layer_idx]
                            block_curve[layer_idx].append(linear_cka(prev_act, current_act))
                            prev_block_acts[layer_idx] = current_act

                if prev_noise is None:
                    prev_noise = noise_pred.detach()
                    if collector is not None:
                        prev_block_acts = dict(collector.activations)
                else:
                    prev_noise = noise_pred.detach()

        finally:
            for handle in handles:
                handle.remove()

        safe_name = action_label.replace("/", "_").replace("+", "plus")
        meta = {
            "action_label": action_label,
            "action_id": action_id,
            "num_steps": len(timesteps),
            "num_pairs": len(step_curve),
        }
        save_curve(output_dir, f"cka_steps_{safe_name}", step_curve, meta)
        try:
            plot_curve(step_curve, f"CKA Step-to-Step - {action_label}", output_dir / f"cka_steps_{safe_name}.png")
        except ImportError:
            pass

        if not args.skip_blocks:
            block_matrix = np.stack([block_curve[layer_idx] for layer_idx in layers], axis=0)
            save_block_curve(output_dir, f"cka_blocks_{safe_name}", block_matrix, layers, meta)

    print(f"Saved CKA step curves to {output_dir}")


if __name__ == "__main__":
    main()
