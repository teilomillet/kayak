"""CLM projection architecture, adapted from Contrastive-LM/CLM (Apache-2.0).

Upstream revision: 7956937c58ed5839c06ddc4dc6b6b61c3a3e4094, src/clm/heads.py.
Kayak changes: explicit configuration validation, restricted checkpoint loading,
finite-value checks, and immutable lifetime instead of hot reload/downloads.
"""

from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
from torch import nn

from .errors import ModelLoadError


class HeadOptions(TypedDict):
    hidden: int
    width: int
    depth: int
    projection_dim: int
    activation: Literal["gelu", "relu", "silu"]
    layernorm: bool
    residual: bool


class ProjectionHead(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        width: int,
        depth: int,
        projection_dim: int,
        activation: Literal["gelu", "relu", "silu"],
        layernorm: bool,
        residual: bool,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(hidden, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
        self.norms = nn.ModuleList(
            nn.LayerNorm(width) if layernorm else nn.Identity() for _ in range(depth - 2)
        )
        self.out = nn.Linear(width, projection_dim)
        self.act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.inp(x))
        for layer, norm in zip(self.hidden, self.norms, strict=True):
            value: torch.Tensor = self.act(norm(layer(x)))
            x = x + value if self.residual else value
        return cast(torch.Tensor, self.out(x))


def load_heads(
    path: Path, hidden_size: int, device: str, encoder_id: str
) -> tuple[ProjectionHead, ProjectionHead, float]:
    try:
        checkpoint: dict[str, object] = torch.load(path, map_location="cpu", weights_only=True)
        cfg = cast(dict[str, object], checkpoint["cfg"])
        if "model" in cfg and cfg["model"] != encoder_id:
            raise ValueError("checkpoint names a different encoder")
        options: dict[str, object] = dict(
            hidden=cfg.get("hidden_size", 4096),
            width=cfg["width"],
            depth=cfg["depth"],
            projection_dim=checkpoint.get("projection_dim", cfg.get("projection_dim", 512)),
            activation=cfg.get("activation", "gelu"),
            layernorm=cfg.get("layernorm", False),
            residual=cfg.get("residual", False),
        )
        for key in ("hidden", "width", "depth", "projection_dim"):
            value = options[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"invalid head configuration: {key}")
        if options["hidden"] != hidden_size or cast(int, options["depth"]) < 2:
            raise ValueError("head input dimension or depth is incompatible")
        if options["activation"] not in {"gelu", "relu", "silu"} or any(
            type(options[k]) is not bool for k in ("layernorm", "residual")
        ):
            raise ValueError("unsupported activation or head configuration")
        # All fields were validated above; preserve the checkpoint's existing recipe.
        head_options = cast(HeadOptions, options)
        heads: list[ProjectionHead] = []
        for key in ("state_head", "action_head"):
            weights = cast(dict[str, torch.Tensor], checkpoint[key])
            if not weights or not all(
                isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in weights.values()
            ):
                raise ValueError("head weights must be finite tensors")
            head = ProjectionHead(**head_options)
            head.load_state_dict(weights, strict=True)
            heads.append(head.eval().requires_grad_(False).to(device=device, dtype=torch.float32))
        logit_scale = torch.as_tensor(checkpoint["logit_scale"], dtype=torch.float32)
        if logit_scale.numel() != 1 or not torch.isfinite(logit_scale).all():
            raise ValueError("logit_scale must be a finite scalar")
        scale = float(logit_scale.exp().clamp(max=100.0).item())
        return heads[0], heads[1], scale
    except Exception as exc:
        raise ModelLoadError(
            f"invalid or incompatible CLM checkpoint ({type(exc).__name__})"
        ) from exc
