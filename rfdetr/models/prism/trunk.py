from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.util.misc import NestedTensor


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [...,4] (cx,cy,w,h) to (x1,y1,x2,y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


@dataclass
class PRISMConfig:
    # profile tokenization
    N: int = 32
    K: int = 8
    delta_ratio: float = 0.35
    thickness_ratio: float = 0.25
    pool_center_ratio: float = 0.25
    # modules
    tcn_layers: int = 2
    tcn_kernel: int = 3
    cross_heads: int = 8
    srm_hidden: int = 128
    pos_dim: int = 32


class _ResTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation, groups=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, C, N]
        # LayerNorm expects last dim, so transpose
        y = x.transpose(1, 2)  # [M, N, C]
        y = self.net[0](y)
        y = y.transpose(1, 2)
        y = self.net[1](y)
        y = self.net[2](y)
        y = self.net[3](y)
        return x + y


class PRISMTrunk(nn.Module):
    """PRISM Trunk (auxiliary branch).

    Training-time: sample profiles from srcs[0] (P4 in this repo) using GT HBB
    boxes (normalized cxcywh), then produce an object embedding of size hidden_dim.

    The embedding can be aligned to matched decoder query features via a
    Hungarian-aligned distillation loss.
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 2, cfg: Optional[PRISMConfig] = None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.cfg = cfg or PRISMConfig()

        # Axial context encoding (TCN) operates on flattened 3-band features
        flat_c = 3 * self.hidden_dim
        self.tcn = nn.ModuleList(
            [_ResTCNBlock(flat_c, kernel=self.cfg.tcn_kernel, dilation=2**i) for i in range(self.cfg.tcn_layers)]
        )

        # Cross-band interaction: center queries attend to side sequences
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.cfg.cross_heads, batch_first=True
        )
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        # SRM (shoreline role modulation)
        srm_in = 4 * self.hidden_dim
        self.srm_s = nn.Sequential(
            nn.Linear(srm_in, self.cfg.srm_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.srm_hidden, self.cfg.srm_hidden),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(1, self.cfg.pos_dim),
            nn.GELU(),
            nn.Linear(self.cfg.pos_dim, self.cfg.pos_dim),
        )
        mod_in = self.hidden_dim + self.cfg.srm_hidden + self.cfg.pos_dim
        self.mod_cont = nn.Sequential(
            nn.Linear(mod_in, 2 * self.hidden_dim),
        )
        self.mod_end = nn.Sequential(
            nn.Linear(mod_in, 2 * self.hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(mod_in, 1),
        )
        self.ln_c = nn.LayerNorm(self.hidden_dim)

        # classification head for optional PRISM CE supervision
        self.cls_head = nn.Linear(self.hidden_dim, self.num_classes)

    @staticmethod
    def _box_abs_xyxy(boxes_cxcywh_norm: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
        """boxes: [M,4] normalized cxcywh, size_hw: [2] (H,W) in pixels."""
        h, w = size_hw[0].float(), size_hw[1].float()
        boxes_abs = boxes_cxcywh_norm.clone()
        boxes_abs[:, 0] = boxes_abs[:, 0] * w
        boxes_abs[:, 2] = boxes_abs[:, 2] * w
        boxes_abs[:, 1] = boxes_abs[:, 1] * h
        boxes_abs[:, 3] = boxes_abs[:, 3] * h
        return cxcywh_to_xyxy(boxes_abs)

    def _sample_one_box(
        self,
        src_b: torch.Tensor,
        img_hw: Tuple[int, int],
        box_xyxy: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a 3-band axial profile for one GT box.

        Args:
            src_b: [1, C, Hf, Wf]
            img_hw: (Himg, Wimg) in pixels (same coord system as box)
            box_xyxy: [4] in pixel coordinates

        Returns:
            Z: [N, 3, C]
        """
        device = src_b.device
        dtype = src_b.dtype
        Himg, Wimg = img_hw
        _, C, Hf, Wf = src_b.shape

        x1, y1, x2, y2 = box_xyxy.to(device=device, dtype=dtype)
        bw = (x2 - x1).clamp(min=1.0)
        bh = (y2 - y1).clamp(min=1.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        c0 = torch.stack([cx, cy], dim=0)  # [2]

        # diagonal direction prior
        u = torch.stack([bw, bh], dim=0)
        u = u / (u.norm() + 1e-6)
        n = torch.stack([-u[1], u[0]], dim=0)

        # axial length (diagonal)
        L = torch.sqrt(bw * bw + bh * bh)

        N = int(self.cfg.N)
        K = int(self.cfg.K)
        t = torch.linspace(-0.5, 0.5, N, device=device, dtype=dtype)
        t = (t + 0.5 / N) * L  # center-of-bin sampling
        t = t - 0.5 * L

        # band offsets and thickness along normal
        short = torch.minimum(bw, bh)
        delta = self.cfg.delta_ratio * short
        a = self.cfg.thickness_ratio * short
        lam = torch.linspace(-a, a, K, device=device, dtype=dtype)
        # combine (band, point) into one axis of length 3K
        offsets = torch.cat(
            [
                (0.0 * delta + lam),
                (-delta + lam),
                (delta + lam),
            ],
            dim=0,
        )  # [3K]

        # grid in pixel coords: [N, 3K, 2]
        # x = c0 + t*u + offsets*n
        grid_xy = c0[None, None, :] + t[:, None, None] * u[None, None, :] + offsets[None, :, None] * n[None, None, :]

        # map to feature coords and then to [-1,1]
        # align_corners=True mapping
        x = grid_xy[..., 0].clamp(0, Wimg - 1)
        y = grid_xy[..., 1].clamp(0, Himg - 1)
        xf = x / max(Wimg - 1, 1) * (Wf - 1)
        yf = y / max(Himg - 1, 1) * (Hf - 1)
        gx = 2.0 * (xf / max(Wf - 1, 1)) - 1.0
        gy = 2.0 * (yf / max(Hf - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1,N,3K,2]

        samp = F.grid_sample(src_b, grid, mode="bilinear", align_corners=True)  # [1,C,N,3K]
        samp = samp.view(1, C, N, 3, K).mean(dim=-1)  # [1,C,N,3]
        Z = samp.permute(0, 2, 3, 1).contiguous().squeeze(0)  # [N,3,C]
        return Z

    def forward(
        self,
        src: torch.Tensor,
        samples: NestedTensor,
        targets: List[Dict],
    ) -> Dict[str, List[torch.Tensor]]:
        """Compute PRISM embeddings for GT boxes.

        Args:
            src: srcs[0], shape [B, C(=hidden_dim), Hf, Wf]
            samples: NestedTensor with padded images
            targets: list of DETR targets, each containing
                - boxes: [Mi,4] normalized cxcywh
                - labels: [Mi]
                - size: [2] (H,W) in pixels (after resize/pad)

        Returns:
            dict with lists (per batch element):
                prism_seq:  [Mi,N,3,C]
                prism_embed:[Mi,C]
                prism_logits:[Mi,num_classes]
        """
        B, C, Hf, Wf = src.shape
        assert C == self.hidden_dim, f"PRISM expects C={self.hidden_dim}, got {C}"

        out_seq: List[torch.Tensor] = []
        out_embed: List[torch.Tensor] = []
        out_logits: List[torch.Tensor] = []

        for b in range(B):
            tgt = targets[b]
            boxes = tgt.get("boxes")
            labels = tgt.get("labels")
            if boxes is None or boxes.numel() == 0:
                out_seq.append(src.new_zeros((0, self.cfg.N, 3, self.hidden_dim)))
                out_embed.append(src.new_zeros((0, self.hidden_dim)))
                out_logits.append(src.new_zeros((0, self.num_classes)))
                continue

            size_hw = tgt.get("size")
            if size_hw is None:
                # fallback to padded tensor shape
                size_hw = torch.as_tensor(samples.tensors.shape[-2:], device=src.device)
            Himg, Wimg = int(size_hw[0].item()), int(size_hw[1].item())

            boxes_xyxy = self._box_abs_xyxy(boxes, size_hw)  # [Mi,4]
            src_b = src[b : b + 1]

            # sample all boxes
            zs = []
            for m in range(boxes_xyxy.shape[0]):
                Z = self._sample_one_box(src_b, (Himg, Wimg), boxes_xyxy[m])  # [N,3,C]
                zs.append(Z)
            Zs = torch.stack(zs, dim=0)  # [Mi,N,3,C]

            # Axial context encoding (TCN)
            Y = Zs.reshape(Zs.shape[0], Zs.shape[1], -1).transpose(1, 2)  # [Mi,3C,N]
            for blk in self.tcn:
                Y = blk(Y)
            Zs = Y.transpose(1, 2).reshape(Zs.shape[0], Zs.shape[1], 3, self.hidden_dim)

            # Cross-band interaction
            center = Zs[:, :, 0, :]  # [Mi,N,C]
            side = torch.cat([Zs[:, :, 1, :], Zs[:, :, 2, :]], dim=1)  # [Mi,2N,C]
            attn_out, _ = self.cross_attn(center, side, side, need_weights=False)
            center = center + attn_out
            center = center + self.cross_ffn(center)

            # SRM
            left = Zs[:, :, 1, :]
            right = Zs[:, :, 2, :]
            s = torch.cat([left, right, left - right, left + right], dim=-1)
            s = self.srm_s(s)  # [Mi,N,Ds]
            # pos encoding
            N = Zs.shape[1]
            pos = torch.linspace(0, 1, N, device=src.device, dtype=src.dtype).view(1, N, 1)
            pos = (pos - 0.5).abs() * 2.0  # 0 at center, 1 at ends
            pos = pos.expand(Zs.shape[0], -1, -1)
            p = self.pos_embed(pos)  # [Mi,N,P]

            c_ln = self.ln_c(center)
            mod_in = torch.cat([c_ln, s, p], dim=-1)
            gamma_beta_cont = self.mod_cont(mod_in)
            gamma_beta_end = self.mod_end(mod_in)
            g_cont, b_cont = gamma_beta_cont.chunk(2, dim=-1)
            g_end, b_end = gamma_beta_end.chunk(2, dim=-1)

            v_cont = (1.0 + g_cont) * c_ln + b_cont
            v_end = (1.0 + g_end) * c_ln + b_end
            alpha = torch.sigmoid(self.gate(mod_in))  # [Mi,N,1]
            center_hat = (1.0 - alpha) * v_cont + alpha * v_end

            Zs_out = Zs.clone()
            Zs_out[:, :, 0, :] = center_hat

            # center-mean pooling over a central window and 3 bands
            mid = N // 2
            half = max(1, int(N * self.cfg.pool_center_ratio / 2))
            lo = max(0, mid - half)
            hi = min(N, mid + half + 1)
            pooled = Zs_out[:, lo:hi, :, :].mean(dim=(1, 2))  # [Mi,C]

            logits = self.cls_head(pooled)

            out_seq.append(Zs_out)
            out_embed.append(pooled)
            out_logits.append(logits)

        return {"prism_seq": out_seq, "prism_embed": out_embed, "prism_logits": out_logits}
