"""
Lightweight SmoothQuant-style fake quantization utilities for HY-WorldPlay.

This mirrors the Matrix-Game SmoothQuant setup:
- weights are statically quantized with per-channel or per-tensor absmax
- activations are quantized per token on the fly
- specific layers can be skipped or assigned custom bitwidths via layer_bits
"""

import importlib
import os
import sys
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5) / q_max
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5) / q_max
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    orig_dtype = t.dtype
    t_flat = t.view(-1, t.shape[-1])
    scales = t_flat.abs().max(dim=-1, keepdim=True).values
    q_max = 2 ** (n_bits - 1) - 1
    scales = (scales / q_max).clamp(min=1e-5)
    t_q = torch.round(t_flat / scales) * scales
    return t_q.view_as(t).to(orig_dtype)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5) / q_max
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8Linear(nn.Module):
    """
    Drop-in Linear replacement that applies activation quantization on the fly
    and stores a statically quantized weight copy.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_quant: str = "per_token",
        quantize_output: bool = False,
        n_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits

        self.register_buffer(
            "weight",
            torch.randn(
                out_features,
                in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((1, out_features), dtype=torch.float16, requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.n_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.n_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x = self.act_quant(x)
        y = F.linear(q_x, self.weight, self.bias)
        return self.output_quant(y)

    @staticmethod
    def from_float(
        module: nn.Linear,
        weight_quant: str = "per_channel",
        act_quant: str = "per_token",
        quantize_output: bool = False,
        n_bits: int = 8,
    ) -> "W8A8Linear":
        assert isinstance(module, nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            n_bits=n_bits,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(module.weight, n_bits=n_bits)
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(module.weight, n_bits=n_bits)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self) -> str:
        bias_str = self.bias is not None
        return (
            f"W8A8Linear({self.in_features}, {self.out_features}, bias={bias_str}, "
            f"weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, "
            f"output_quant={self.output_quant_name}, n_bits={self.n_bits})"
        )


_VIDITQ_ROTATION_CACHE: Dict[Tuple[int, str], Optional[torch.Tensor]] = {}


def _load_viditq_random_hadamard_matrix():
    try:
        module = importlib.import_module("qdiff.quarot.quarot_utils")
        return module.random_hadamard_matrix
    except Exception:
        quant_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ViDiT-Q", "quant_utils"))
        if quant_root not in sys.path:
            sys.path.append(quant_root)
        module = importlib.import_module("qdiff.quarot.quarot_utils")
        return module.random_hadamard_matrix


def _get_viditq_rotation_matrix(in_features: int, device: torch.device) -> Optional[torch.Tensor]:
    key = (in_features, str(device))
    if key not in _VIDITQ_ROTATION_CACHE:
        random_hadamard_matrix = _load_viditq_random_hadamard_matrix()
        try:
            _VIDITQ_ROTATION_CACHE[key] = random_hadamard_matrix(in_features, str(device)).float()
        except Exception:
            _VIDITQ_ROTATION_CACHE[key] = None
    return _VIDITQ_ROTATION_CACHE[key]


class ViDiTQLinear(nn.Module):
    """
    ViDiT-Q style linear layer with scale-based + rotation-based quantization.

    This mirrors the ViDiT-Q design but uses the same bitwidth for weights and activations
    (consistent with W8A8Linear) and supports 2D or 3D inputs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_quant: str = "per_token",
        n_bits: int = 8,
        alpha: float = 0.75,
        auto_calibrate: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.alpha = alpha
        self.auto_calibrate = auto_calibrate

        self.register_buffer(
            "weight",
            torch.randn(
                out_features,
                in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((1, out_features), dtype=torch.float16, requires_grad=False),
            )
        else:
            self.register_buffer("bias", None)

        self.fp_weight = None
        self.channel_mask = None
        self.rotation_matrix = None

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.n_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.n_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

    def _compute_channel_mask(self, act_mask: torch.Tensor, weight_mask: torch.Tensor) -> torch.Tensor:
        if weight_mask.device != act_mask.device:
            weight_mask = weight_mask.to(act_mask.device)
        act_mask = act_mask.abs().clamp(min=1e-3)
        weight_mask = weight_mask.abs()
        channel_mask = (weight_mask ** self.alpha) / (act_mask ** (1 - self.alpha))
        return channel_mask.clamp(min=1e-5)

    def _ensure_rotation(self, device: torch.device) -> None:
        if self.rotation_matrix is None or self.rotation_matrix.device != device:
            self.rotation_matrix = _get_viditq_rotation_matrix(self.in_features, device=device)

    def _update_quantized_weight(self) -> None:
        if self.fp_weight is None:
            raise RuntimeError("ViDiTQLinear requires fp_weight to update quantized weights.")
        if self.channel_mask is None:
            raise RuntimeError("ViDiTQLinear requires channel_mask to update quantized weights.")
        target_device = self.channel_mask.device
        self._ensure_rotation(target_device)

        w = self.fp_weight.detach().to(device=target_device, dtype=torch.float32)
        channel_mask = self.channel_mask.to(device=target_device)
        w = w / channel_mask.reshape([1, -1])
        if hasattr(self, "weight_quant_name") and self.weight_quant_name == "per_tensor":
            w = quantize_weight_per_tensor_absmax(w, n_bits=self.n_bits)
        else:
            w = quantize_weight_per_channel_absmax(w, n_bits=self.n_bits)
        if self.rotation_matrix is not None:
            w = torch.matmul(w, self.rotation_matrix.float())
        if hasattr(self, "weight_quant_name") and self.weight_quant_name == "per_tensor":
            w = quantize_weight_per_tensor_absmax(w, n_bits=self.n_bits)
        else:
            w = quantize_weight_per_channel_absmax(w, n_bits=self.n_bits)
        self.weight = w.to(device=target_device, dtype=self.weight.dtype)

    def _maybe_init_from_input(self, x: torch.Tensor) -> None:
        if self.channel_mask is not None or not self.auto_calibrate:
            return
        x_flat = x.view(-1, x.shape[-1])
        act_mask = x_flat.abs().max(dim=0).values
        weight_mask = self.fp_weight.detach().to(act_mask.device).abs().max(dim=0)[0]
        self.channel_mask = self._compute_channel_mask(act_mask, weight_mask)
        self._update_quantized_weight()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        if self.fp_weight is not None:
            self.fp_weight = self.fp_weight.to(*args, **kwargs)
        if self.channel_mask is not None:
            self.channel_mask = self.channel_mask.to(*args, **kwargs)
        if self.rotation_matrix is not None:
            self.rotation_matrix = self.rotation_matrix.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_mask is None:
            self._maybe_init_from_input(x)

        if self.channel_mask is not None:
            x = x * self.channel_mask.reshape([1] * (x.dim() - 1) + [-1])
        if self.rotation_matrix is not None:
            x = torch.matmul(x.float(), self.rotation_matrix.float()).to(dtype=x.dtype)
        q_x = self.act_quant(x)
        return F.linear(q_x, self.weight, self.bias)

    @staticmethod
    def from_float(
        module: nn.Linear,
        weight_quant: str = "per_channel",
        act_quant: str = "per_token",
        n_bits: int = 8,
        alpha: float = 0.75,
        act_mask: Optional[torch.Tensor] = None,
        auto_calibrate: bool = True,
    ) -> "ViDiTQLinear":
        assert isinstance(module, nn.Linear)
        new_module = ViDiTQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            n_bits=n_bits,
            alpha=alpha,
            auto_calibrate=auto_calibrate,
        )
        new_module.fp_weight = module.weight.detach().clone()
        new_module.weight_quant_name = weight_quant

        if module.bias is not None:
            new_module.bias = module.bias

        if act_mask is not None:
            act_mask = act_mask.to(new_module.fp_weight.device)
            weight_mask = new_module.fp_weight.abs().max(dim=0)[0]
            new_module.channel_mask = new_module._compute_channel_mask(act_mask, weight_mask)
            new_module._update_quantized_weight()
        else:
            if weight_quant == "per_channel":
                new_module.weight = quantize_weight_per_channel_absmax(
                    new_module.fp_weight.detach().clone(), n_bits=n_bits
                )
            elif weight_quant == "per_tensor":
                new_module.weight = quantize_weight_per_tensor_absmax(
                    new_module.fp_weight.detach().clone(), n_bits=n_bits
                )
            else:
                raise ValueError(f"Invalid weight_quant: {weight_quant}")

        return new_module

    def __repr__(self) -> str:
        bias_str = self.bias is not None
        return (
            f"ViDiTQLinear({self.in_features}, {self.out_features}, bias={bias_str}, "
            f"weight_quant={getattr(self, 'weight_quant_name', 'unknown')}, "
            f"act_quant={self.act_quant_name}, n_bits={self.n_bits}, alpha={self.alpha})"
        )


def quantize_hyworldplay(
    model: nn.Module,
    weight_quant: str = "per_channel",
    act_quant: str = "per_token",
    double_n_bits: Optional[int] = 4,
    single_n_bits: Optional[int] = 4,
    final_n_bits: Optional[int] = None,
    cond_n_bits: Optional[int] = None,
    layer_bits: Optional[Dict[str, Optional[int]]] = None,
) -> nn.Module:
    """
    Apply SmoothQuant-style fake quantization to the HY-WorldPlay transformer.

    Args:
        model: Pipeline or transformer instance (expects .transformer or block lists).
        weight_quant: Weight quantization granularity ("per_channel" or "per_tensor").
        act_quant: Activation quantization granularity ("per_token" or "per_tensor").
        double_n_bits: Default bitwidth for MMDoubleStreamBlock projections/MLPs.
        single_n_bits: Default bitwidth for MMSingleStreamBlock projections/MLPs.
        final_n_bits: Optional bitwidth for the FinalLayer projection (skip if None).
        cond_n_bits: Optional bitwidth for time/action/vector embedders (skip if None).
        layer_bits: Optional overrides mapping layer names -> bitwidth or None to skip.
            Supported names (globally) and optional per-block overrides with ".<idx>":
              double.img_attn.q/k/v/proj
              double.img_mod.linear
              double.txt_mod.linear
              double.txt_attn.q/k/v/proj
              double.img_mlp.fc1/fc2
              double.txt_mlp.fc1/fc2
              double.img_attn.prope_proj
              single.attn.q/k/v
              single.mlp.fc1
              single.out
              final.linear
              cond.time_in.fc1/fc2
              cond.time_r_in.fc1/fc2
              cond.vector_in.in/out
              cond.action_in.fc1/fc2
    """
    transformer = getattr(model, "transformer", model)
    if transformer is None:
        raise ValueError("Could not locate transformer on the provided model.")

    def _resolve_bits(name: str, idx: Optional[int], default: Optional[int]) -> Optional[int]:
        if layer_bits:
            candidates = [name]
            if idx is not None:
                candidates.insert(0, f"{name}.{idx}")
            for key in candidates:
                if key in layer_bits:
                    return layer_bits[key]
        return default

    def _quantize_linear_attr(parent: nn.Module, attr_name: str, name: str, idx: Optional[int], default_bits: Optional[int]) -> None:
        bits = _resolve_bits(name, idx, default_bits)
        if bits is None or not hasattr(parent, attr_name):
            return
        current = getattr(parent, attr_name)
        if isinstance(current, W8A8Linear) or not isinstance(current, nn.Linear):
            return
        setattr(
            parent,
            attr_name,
            W8A8Linear.from_float(current, weight_quant=weight_quant, act_quant=act_quant, n_bits=bits),
        )

    def _quantize_mlp(mlp: nn.Module, name: str, idx: Optional[int], default_bits: Optional[int]) -> None:
        if mlp is None or not hasattr(mlp, "fc1") or not hasattr(mlp, "fc2"):
            return
        _quantize_linear_attr(mlp, "fc1", f"{name}.fc1", idx, default_bits)
        _quantize_linear_attr(mlp, "fc2", f"{name}.fc2", idx, default_bits)

    def _quantize_seq_linear(seq: nn.Sequential, linear_idx: int, name: str, idx: Optional[int], default_bits: Optional[int]) -> None:
        if seq is None or not isinstance(seq, nn.Sequential) or linear_idx >= len(seq):
            return
        if not isinstance(seq[linear_idx], nn.Linear):
            return
        bits = _resolve_bits(name, idx, default_bits)
        if bits is None:
            return
        seq[linear_idx] = W8A8Linear.from_float(seq[linear_idx], weight_quant=weight_quant, act_quant=act_quant, n_bits=bits)

    def _quantize_conv_weight(conv: nn.Module, name: str, idx: Optional[int], default_bits: Optional[int]) -> None:
        bits = _resolve_bits(name, idx, default_bits)
        if bits is None or conv is None or not isinstance(conv, (nn.Conv2d, nn.Conv3d)):
            return
        quantize_weight_per_tensor_absmax(conv.weight, n_bits=bits)
        if conv.bias is not None:
            quantize_weight_per_tensor_absmax(conv.bias, n_bits=bits)

    # Double-stream blocks
    for idx, block in enumerate(getattr(transformer, "double_blocks", [])):
        block_bits = _resolve_bits("double", idx, double_n_bits)
        if block_bits is None:
            continue

        _quantize_linear_attr(block, "img_attn_q", "double.img_attn.q", idx, block_bits)
        _quantize_linear_attr(block, "img_attn_k", "double.img_attn.k", idx, block_bits)
        _quantize_linear_attr(block, "img_attn_v", "double.img_attn.v", idx, block_bits)
        _quantize_linear_attr(block, "img_attn_proj", "double.img_attn.proj", idx, block_bits)
        _quantize_linear_attr(getattr(block, "img_mod", None), "linear", "double.img_mod.linear", idx, block_bits)

        _quantize_linear_attr(block, "txt_attn_q", "double.txt_attn.q", idx, block_bits)
        _quantize_linear_attr(block, "txt_attn_k", "double.txt_attn.k", idx, block_bits)
        _quantize_linear_attr(block, "txt_attn_v", "double.txt_attn.v", idx, block_bits)
        _quantize_linear_attr(block, "txt_attn_proj", "double.txt_attn.proj", idx, block_bits)
        _quantize_linear_attr(getattr(block, "txt_mod", None), "linear", "double.txt_mod.linear", idx, block_bits)

        _quantize_mlp(getattr(block, "img_mlp", None), "double.img_mlp", idx, block_bits)
        _quantize_mlp(getattr(block, "txt_mlp", None), "double.txt_mlp", idx, block_bits)

        if hasattr(block, "img_attn_prope_proj"):
            _quantize_linear_attr(block, "img_attn_prope_proj", "double.img_attn.prope_proj", idx, block_bits)

    # Single-stream blocks
    for idx, block in enumerate(getattr(transformer, "single_blocks", [])):
        block_bits = _resolve_bits("single", idx, single_n_bits)
        if block_bits is None:
            continue

        _quantize_linear_attr(block, "linear1_q", "single.attn.q", idx, block_bits)
        _quantize_linear_attr(block, "linear1_k", "single.attn.k", idx, block_bits)
        _quantize_linear_attr(block, "linear1_v", "single.attn.v", idx, block_bits)
        _quantize_linear_attr(block, "linear1_mlp", "single.mlp.fc1", idx, block_bits)

        if hasattr(block, "linear2"):
            _quantize_linear_attr(block.linear2, "fc", "single.out", idx, block_bits)

    # Final output projection
    if hasattr(transformer, "final_layer"):
        _quantize_linear_attr(
            transformer.final_layer,
            "linear",
            "final.linear",
            None,
            final_n_bits,
        )
        if hasattr(transformer.final_layer, "adaLN_modulation"):
            _quantize_seq_linear(
                transformer.final_layer.adaLN_modulation,
                1,
                "final.adaln",
                None,
                final_n_bits,
            )

    # Time/action/vector embedders and other front-end projections
    if cond_n_bits is not None:
        def _quantize_embedder(embedder: nn.Module, name: str) -> None:
            if embedder is None or not hasattr(embedder, "mlp"):
                return
            mlp = embedder.mlp
            if isinstance(mlp, nn.Sequential) and len(mlp) >= 3:
                first = _resolve_bits(f"{name}.fc1", None, cond_n_bits)
                last = _resolve_bits(f"{name}.fc2", None, cond_n_bits)
                if first is not None and isinstance(mlp[0], nn.Linear):
                    mlp[0] = W8A8Linear.from_float(mlp[0], weight_quant=weight_quant, act_quant=act_quant, n_bits=first)
                if last is not None and isinstance(mlp[2], nn.Linear):
                    mlp[2] = W8A8Linear.from_float(mlp[2], weight_quant=weight_quant, act_quant=act_quant, n_bits=last)

        def _quantize_mlp_embedder(embedder: nn.Module, name: str) -> None:
            if embedder is None:
                return
            _quantize_linear_attr(embedder, "in_layer", f"{name}.in", None, cond_n_bits)
            _quantize_linear_attr(embedder, "out_layer", f"{name}.out", None, cond_n_bits)

        _quantize_embedder(getattr(transformer, "time_in", None), "cond.time_in")
        _quantize_embedder(getattr(transformer, "time_r_in", None), "cond.time_r_in")
        _quantize_embedder(getattr(transformer, "action_in", None), "cond.action_in")
        _quantize_mlp_embedder(getattr(transformer, "vector_in", None), "cond.vector_in")

        # PatchEmbed conv projection (img_in.proj)
        _quantize_conv_weight(getattr(transformer, "img_in", None).proj if hasattr(getattr(transformer, "img_in", None), "proj") else None,
                              "img_in.proj", None, cond_n_bits)

        # Text projection path
        txt_in = getattr(transformer, "txt_in", None)
        if txt_in is not None:
            if hasattr(txt_in, "input_embedder"):  # SingleTokenRefiner path
                _quantize_linear_attr(txt_in, "input_embedder", "txt_in.input_embedder", None, cond_n_bits)
                if hasattr(txt_in, "c_embedder"):
                    _quantize_linear_attr(txt_in.c_embedder, "linear_1", "txt_in.c_embedder.linear1", None, cond_n_bits)
                    _quantize_linear_attr(txt_in.c_embedder, "linear_2", "txt_in.c_embedder.linear2", None, cond_n_bits)
                if hasattr(txt_in, "t_embedder"):
                    _quantize_embedder(txt_in.t_embedder, "txt_in.t_embedder")
                if hasattr(txt_in, "individual_token_refiner"):
                    refiner = txt_in.individual_token_refiner
                    if hasattr(refiner, "blocks"):
                        for b_idx, block in enumerate(refiner.blocks):
                            _quantize_linear_attr(block, "self_attn_qkv", "txt_in.refiner.self_attn_qkv", b_idx, cond_n_bits)
                            _quantize_linear_attr(block, "self_attn_proj", "txt_in.refiner.self_attn_proj", b_idx, cond_n_bits)
                            if hasattr(block, "mlp"):
                                _quantize_mlp(block.mlp, "txt_in.refiner.mlp", b_idx, cond_n_bits)
                            if hasattr(block, "adaLN_modulation") and isinstance(block.adaLN_modulation, nn.Sequential) and len(block.adaLN_modulation) >= 2:
                                _quantize_seq_linear(block.adaLN_modulation, 1, "txt_in.refiner.adaln", b_idx, cond_n_bits)
            else:
                # Plain TextProjection path
                _quantize_linear_attr(txt_in, "linear_1", "txt_in.linear1", None, cond_n_bits)
                _quantize_linear_attr(txt_in, "linear_2", "txt_in.linear2", None, cond_n_bits)

        # Vision projection
        vision_in = getattr(transformer, "vision_in", None)
        if vision_in is not None and hasattr(vision_in, "proj"):
            _quantize_seq_linear(vision_in.proj, 1, "vision_in.proj.fc1", None, cond_n_bits)
            _quantize_seq_linear(vision_in.proj, 3, "vision_in.proj.fc2", None, cond_n_bits)

        # ByT5 mapper
        byt5_in = getattr(transformer, "byt5_in", None)
        if byt5_in is not None:
            _quantize_linear_attr(byt5_in, "fc1", "byt5_in.fc1", None, cond_n_bits)
            _quantize_linear_attr(byt5_in, "fc2", "byt5_in.fc2", None, cond_n_bits)
            _quantize_linear_attr(byt5_in, "fc3", "byt5_in.fc3", None, cond_n_bits)

        # Cond type embedding
        if hasattr(transformer, "cond_type_embedding") and transformer.cond_type_embedding is not None:
            quantize_weight_per_tensor_absmax(transformer.cond_type_embedding.weight, n_bits=cond_n_bits)

    return transformer


def quantize_hyworldplay_viditq(
    model: nn.Module,
    weight_quant: str = "per_channel",
    act_quant: str = "per_token",
    double_n_bits: Optional[int] = None,
    single_n_bits: Optional[int] = None,
    final_n_bits: Optional[int] = None,
    cond_n_bits: Optional[int] = None,
    alpha: float = 0.75,
    layer_bits: Optional[Dict[str, Optional[int]]] = None,
    act_masks: Optional[Dict[str, torch.Tensor]] = None,
    auto_calibrate: bool = True,
) -> nn.Module:
    """
    Apply ViDiT-Q style quantization to the HY-WorldPlay transformer.

    Args:
        model: Pipeline or transformer instance (expects .transformer or block lists).
        weight_quant: Weight quantization granularity ("per_channel" or "per_tensor").
        act_quant: Activation quantization granularity ("per_token" or "per_tensor").
        double_n_bits: Default bitwidth for MMDoubleStreamBlock projections/MLPs.
        single_n_bits: Default bitwidth for MMSingleStreamBlock projections/MLPs.
        final_n_bits: Optional bitwidth for the FinalLayer projection (skip if None).
        cond_n_bits: Optional bitwidth for time/action/vector embedders (skip if None).
        alpha: ViDiT-Q channel balancing exponent.
        layer_bits: Optional overrides mapping layer names -> bitwidth or None to skip.
            Supported names (globally) and optional per-block overrides with ".<idx>":
              double.img_attn.q/k/v/proj
              double.img_mod.linear
              double.txt_attn.q/k/v/proj
              double.txt_mod.linear
              double.img_mlp.fc1/fc2
              double.txt_mlp.fc1/fc2
              double.img_attn.prope_proj
              single.attn.q/k/v
              single.mlp.fc1
              single.out
              single.modulation.linear
              final.linear
              final.adaln
              cond.time_in.fc1/fc2
              cond.time_r_in.fc1/fc2
              cond.vector_in.in/out
              cond.action_in.fc1/fc2
        act_masks: Optional overrides mapping layer names -> activation masks.
        auto_calibrate: If True, initialize channel masks from the first forward pass.
    """
    transformer = getattr(model, "transformer", model)
    if transformer is None:
        raise ValueError("Could not locate transformer on the provided model.")

    def _resolve_bits(name: str, idx: Optional[int], default: Optional[int]) -> Optional[int]:
        if layer_bits:
            candidates = [name]
            if idx is not None:
                candidates.insert(0, f"{name}.{idx}")
            for key in candidates:
                if key in layer_bits:
                    return layer_bits[key]
        return default

    def _resolve_mask(name: str, idx: Optional[int]) -> Optional[torch.Tensor]:
        if act_masks:
            candidates = [name]
            if idx is not None:
                candidates.insert(0, f"{name}.{idx}")
            for key in candidates:
                if key in act_masks:
                    return act_masks[key]
        return None

    def _quantize_linear_attr(
        parent: nn.Module, attr_name: str, name: str, idx: Optional[int], default_bits: Optional[int]
    ) -> None:
        bits = _resolve_bits(name, idx, default_bits)
        if bits is None or not hasattr(parent, attr_name):
            return
        current = getattr(parent, attr_name)
        if isinstance(current, ViDiTQLinear) or not isinstance(current, nn.Linear):
            return
        act_mask = _resolve_mask(name, idx)
        setattr(
            parent,
            attr_name,
            ViDiTQLinear.from_float(
                current,
                weight_quant=weight_quant,
                act_quant=act_quant,
                n_bits=bits,
                alpha=alpha,
                act_mask=act_mask,
                auto_calibrate=auto_calibrate,
            ),
        )

    def _quantize_mlp(mlp: nn.Module, name: str, idx: Optional[int], default_bits: Optional[int]) -> None:
        if mlp is None or not hasattr(mlp, "fc1") or not hasattr(mlp, "fc2"):
            return
        _quantize_linear_attr(mlp, "fc1", f"{name}.fc1", idx, default_bits)
        _quantize_linear_attr(mlp, "fc2", f"{name}.fc2", idx, default_bits)

    def _quantize_seq_linear(
        seq: nn.Sequential, linear_idx: int, name: str, idx: Optional[int], default_bits: Optional[int]
    ) -> None:
        if seq is None or not isinstance(seq, nn.Sequential) or linear_idx >= len(seq):
            return
        if not isinstance(seq[linear_idx], nn.Linear):
            return
        bits = _resolve_bits(name, idx, default_bits)
        if bits is None:
            return
        act_mask = _resolve_mask(name, idx)
        seq[linear_idx] = ViDiTQLinear.from_float(
            seq[linear_idx],
            weight_quant=weight_quant,
            act_quant=act_quant,
            n_bits=bits,
            alpha=alpha,
            act_mask=act_mask,
            auto_calibrate=auto_calibrate,
        )

    def _quantize_conv_weight(conv: nn.Module, name: str, idx: Optional[int], default_bits: Optional[int]) -> None:
        bits = _resolve_bits(name, idx, default_bits)
        if bits is None or conv is None or not isinstance(conv, (nn.Conv2d, nn.Conv3d)):
            return
        quantize_weight_per_tensor_absmax(conv.weight, n_bits=bits)
        if conv.bias is not None:
            quantize_weight_per_tensor_absmax(conv.bias, n_bits=bits)

    for idx, block in enumerate(getattr(transformer, "double_blocks", [])):
        block_bits = _resolve_bits("double", idx, double_n_bits)
        _quantize_linear_attr(block, "img_attn_q", "double.img_attn.q", idx, block_bits)
        _quantize_linear_attr(block, "img_attn_k", "double.img_attn.k", idx, block_bits)
        _quantize_linear_attr(block, "img_attn_v", "double.img_attn.v", idx, block_bits)
        _quantize_linear_attr(block, "img_attn_proj", "double.img_attn.proj", idx, block_bits)
        _quantize_linear_attr(getattr(block, "img_mod", None), "linear", "double.img_mod.linear", idx, block_bits)

        _quantize_linear_attr(block, "txt_attn_q", "double.txt_attn.q", idx, block_bits)
        _quantize_linear_attr(block, "txt_attn_k", "double.txt_attn.k", idx, block_bits)
        _quantize_linear_attr(block, "txt_attn_v", "double.txt_attn.v", idx, block_bits)
        _quantize_linear_attr(block, "txt_attn_proj", "double.txt_attn.proj", idx, block_bits)
        _quantize_linear_attr(getattr(block, "txt_mod", None), "linear", "double.txt_mod.linear", idx, block_bits)

        _quantize_mlp(getattr(block, "img_mlp", None), "double.img_mlp", idx, block_bits)
        _quantize_mlp(getattr(block, "txt_mlp", None), "double.txt_mlp", idx, block_bits)

        if hasattr(block, "img_attn_prope_proj"):
            _quantize_linear_attr(block, "img_attn_prope_proj", "double.img_attn.prope_proj", idx, block_bits)

    for idx, block in enumerate(getattr(transformer, "single_blocks", [])):
        block_bits = _resolve_bits("single", idx, single_n_bits)
        _quantize_linear_attr(block, "linear1_q", "single.attn.q", idx, block_bits)
        _quantize_linear_attr(block, "linear1_k", "single.attn.k", idx, block_bits)
        _quantize_linear_attr(block, "linear1_v", "single.attn.v", idx, block_bits)
        _quantize_linear_attr(block, "linear1_mlp", "single.mlp.fc1", idx, block_bits)
        _quantize_linear_attr(getattr(block, "modulation", None), "linear", "single.modulation.linear", idx, block_bits)

        if hasattr(block, "linear2"):
            _quantize_linear_attr(block.linear2, "fc", "single.out", idx, block_bits)

    if hasattr(transformer, "final_layer"):
        _quantize_linear_attr(
            transformer.final_layer,
            "linear",
            "final.linear",
            None,
            final_n_bits,
        )
        if hasattr(transformer.final_layer, "adaLN_modulation"):
            _quantize_seq_linear(
                transformer.final_layer.adaLN_modulation,
                1,
                "final.adaln",
                None,
                final_n_bits,
            )

    if cond_n_bits is not None:
        def _quantize_embedder(embedder: nn.Module, name: str) -> None:
            if embedder is None or not hasattr(embedder, "mlp"):
                return
            mlp = embedder.mlp
            if isinstance(mlp, nn.Sequential) and len(mlp) >= 3:
                first = _resolve_bits(f"{name}.fc1", None, cond_n_bits)
                last = _resolve_bits(f"{name}.fc2", None, cond_n_bits)
                if first is not None and isinstance(mlp[0], nn.Linear):
                    act_mask = _resolve_mask(f"{name}.fc1", None)
                    mlp[0] = ViDiTQLinear.from_float(
                        mlp[0],
                        weight_quant=weight_quant,
                        act_quant=act_quant,
                        n_bits=first,
                        alpha=alpha,
                        act_mask=act_mask,
                        auto_calibrate=auto_calibrate,
                    )
                if last is not None and isinstance(mlp[2], nn.Linear):
                    act_mask = _resolve_mask(f"{name}.fc2", None)
                    mlp[2] = ViDiTQLinear.from_float(
                        mlp[2],
                        weight_quant=weight_quant,
                        act_quant=act_quant,
                        n_bits=last,
                        alpha=alpha,
                        act_mask=act_mask,
                        auto_calibrate=auto_calibrate,
                    )

        def _quantize_mlp_embedder(embedder: nn.Module, name: str) -> None:
            if embedder is None:
                return
            _quantize_linear_attr(embedder, "in_layer", f"{name}.in", None, cond_n_bits)
            _quantize_linear_attr(embedder, "out_layer", f"{name}.out", None, cond_n_bits)

        _quantize_embedder(getattr(transformer, "time_in", None), "cond.time_in")
        _quantize_embedder(getattr(transformer, "time_r_in", None), "cond.time_r_in")
        _quantize_embedder(getattr(transformer, "action_in", None), "cond.action_in")
        _quantize_mlp_embedder(getattr(transformer, "vector_in", None), "cond.vector_in")

        _quantize_conv_weight(
            getattr(transformer, "img_in", None).proj if hasattr(getattr(transformer, "img_in", None), "proj") else None,
            "img_in.proj",
            None,
            cond_n_bits,
        )

        txt_in = getattr(transformer, "txt_in", None)
        if txt_in is not None:
            if hasattr(txt_in, "input_embedder"):
                _quantize_linear_attr(txt_in, "input_embedder", "txt_in.input_embedder", None, cond_n_bits)
                if hasattr(txt_in, "c_embedder"):
                    _quantize_linear_attr(txt_in.c_embedder, "linear_1", "txt_in.c_embedder.linear1", None, cond_n_bits)
                    _quantize_linear_attr(txt_in.c_embedder, "linear_2", "txt_in.c_embedder.linear2", None, cond_n_bits)
                if hasattr(txt_in, "t_embedder"):
                    _quantize_embedder(txt_in.t_embedder, "txt_in.t_embedder")
                if hasattr(txt_in, "individual_token_refiner"):
                    refiner = txt_in.individual_token_refiner
                    if hasattr(refiner, "blocks"):
                        for b_idx, block in enumerate(refiner.blocks):
                            _quantize_linear_attr(block, "self_attn_qkv", "txt_in.refiner.self_attn_qkv", b_idx, cond_n_bits)
                            _quantize_linear_attr(block, "self_attn_proj", "txt_in.refiner.self_attn_proj", b_idx, cond_n_bits)
                            if hasattr(block, "mlp"):
                                _quantize_mlp(block.mlp, "txt_in.refiner.mlp", b_idx, cond_n_bits)
                            if hasattr(block, "adaLN_modulation") and isinstance(block.adaLN_modulation, nn.Sequential) and len(block.adaLN_modulation) >= 2:
                                _quantize_seq_linear(block.adaLN_modulation, 1, "txt_in.refiner.adaln", b_idx, cond_n_bits)
            else:
                _quantize_linear_attr(txt_in, "linear_1", "txt_in.linear1", None, cond_n_bits)
                _quantize_linear_attr(txt_in, "linear_2", "txt_in.linear2", None, cond_n_bits)

        vision_in = getattr(transformer, "vision_in", None)
        if vision_in is not None and hasattr(vision_in, "proj"):
            _quantize_seq_linear(vision_in.proj, 1, "vision_in.proj.fc1", None, cond_n_bits)
            _quantize_seq_linear(vision_in.proj, 3, "vision_in.proj.fc2", None, cond_n_bits)

        byt5_in = getattr(transformer, "byt5_in", None)
        if byt5_in is not None:
            _quantize_linear_attr(byt5_in, "fc1", "byt5_in.fc1", None, cond_n_bits)
            _quantize_linear_attr(byt5_in, "fc2", "byt5_in.fc2", None, cond_n_bits)
            _quantize_linear_attr(byt5_in, "fc3", "byt5_in.fc3", None, cond_n_bits)

        if hasattr(transformer, "cond_type_embedding") and transformer.cond_type_embedding is not None:
            quantize_weight_per_tensor_absmax(transformer.cond_type_embedding.weight, n_bits=cond_n_bits)

    return transformer
