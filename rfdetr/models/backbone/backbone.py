# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoBackbone
from peft import LoraConfig, get_peft_model, PeftModel

from rfdetr.util.misc import NestedTensor, is_main_process

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""
    def __init__(self,
                 name: str,
                 pretrained_encoder: str=None,
                 window_block_indexes: list=None,
                 drop_path=0.0,
                 out_channels=256,
                 out_feature_indexes: list=None,
                 projector_scale: list=None,
                 use_cls_token: bool = False,
                 freeze_encoder: bool = False,
                 layer_norm: bool = False,
                 target_shape: tuple[int, int] = (640, 640),
                 rms_norm: bool = False,
                 backbone_lora: bool = False,
                 gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = True,
                 patch_size: int = 14,
                 num_windows: int = 4,
                 #positional_encoding_size: bool = False,
                 positional_encoding_size: int = 0,
                 # optional DINOv3 loading knobs (HF/Hub) ----
                 dinov3_repo_dir: str | None = None,
                 dinov3_weights_path: str | None = None,
                 dinov3_hf_token: str | None = None,
                 dinov3_prefer_hf: bool = True,
                 # WSS adapter knobs (for DINOv3)
                 wss_enable: bool = False,
                 wss_layers: list[int] | None = None,
                 wss_levels: int = 1,
                 wss_reduction_dim: int = 64,
                 ):
        super().__init__()

        # Accept either "dinov2_*" (existing) or "dinov3_*" (new).
        name_parts = name.split("_")
        family = name_parts[0]
        size = name_parts[-1]

        if family == "dinov2":
            # Existing semantics: optional "registers" and/or "windowed" tokens. :contentReference[oaicite:7]{index=7}
            use_registers = False
            if "registers" in name_parts:
                use_registers = True
                name_parts.remove("registers")
            use_windowed_attn = False
            if "windowed" in name_parts:
                use_windowed_attn = True
                name_parts.remove("windowed")
            assert len(name_parts) == 2, "name should be dinov2, then either registers, windowed, both, or none, then the size"
            self.encoder = DinoV2(
                size=size,
                out_feature_indexes=out_feature_indexes,
                shape=target_shape,
                use_registers=use_registers,
                use_windowed_attn=use_windowed_attn,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights,
                patch_size=patch_size,
                num_windows=num_windows,
                positional_encoding_size=positional_encoding_size,
            )
        elif family == "dinov3":
            # new DINOv3 branch (no registers/windowing here)
            self.encoder = DinoV3(
                size=size,
                out_feature_indexes=out_feature_indexes,
                shape=target_shape,
                patch_size=patch_size if patch_size else 16,
                # reuse your existing flag for "load pretrained?" to avoid config churn
                load_dinov3_weights=load_dinov2_weights,
                hf_token=dinov3_hf_token,
                repo_dir=dinov3_repo_dir,
                weights=dinov3_weights_path,
                prefer_hf=dinov3_prefer_hf,
                wss_enable=wss_enable,
                wss_layers=wss_layers,
                wss_levels=wss_levels,
                wss_reduction_dim=wss_reduction_dim,
            )
        else:
            raise AssertionError(f"Backbone name must start with 'dinov2' or 'dinov3', got: {family}")
        # build encoder + projector as backbone module
        if freeze_encoder:
            # Freeze backbone parameters while keeping optional adapters trainable.
            if hasattr(self.encoder, "freeze_base") and callable(getattr(self.encoder, "freeze_base")):
                self.encoder.freeze_base()
            else:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False
                # 初始化DINOv3特征导出开关
        self._init_dump_flags()




        # ===== Dump DINOv3中间特征的小工具 =====
    def _init_dump_flags(self):
        import os
        self._dump_feats = os.environ.get("DUMP_DINOV3_FEATS", "0") == "1"
        self._dump_dir = os.environ.get("DUMP_DINOV3_DIR", os.path.join(os.getcwd(), "dinov3_feats"))
        self._dump_max = int(os.environ.get("DUMP_DINOV3_MAX", "50"))
        self._dump_per_batch = int(os.environ.get("DUMP_DINOV3_PERBATCH", "2"))
        self._dump_save_pt = os.environ.get("DUMP_DINOV3_PT", "0") == "1"

        # 美观控制
        self._dump_style = os.environ.get("DUMP_STYLE", "overlay").lower()  # overlay | pca | texture
        self._dump_alpha = float(os.environ.get("DUMP_ALPHA", "0.5"))        # 叠加透明度
        self._dump_upsample = os.environ.get("DUMP_UPSAMPLE", "orig").lower()# orig | feat
        self._dump_colormap = os.environ.get("DUMP_COLORMAP", "turbo").lower()
        self._dump_texture_blend = float(os.environ.get("DUMP_TEXTURE_BLEND", "0.5"))

        self._dump_count = 0
        if self._dump_feats:
            os.makedirs(self._dump_dir, exist_ok=True)
            print(f"[DUMP] enabled → {self._dump_dir}  style={self._dump_style}, alpha={self._dump_alpha}")

    def _maybe_dump(self, feats, tag: str, imgs: 'torch.Tensor|None'=None):
        """
        feats: List[Tensor], each [B, C, Hf, Wf]
        imgs : Optional[Tensor] = input batch images [B, 3, H, W] (normalized) — 用于叠加到原图
        输出文件命名: {count:06d}_{tag}_{Lidx}_{Bidx}.png
        """
        import os, numpy as np, torch
        import torch.nn.functional as F
        from PIL import Image

        if not getattr(self, "_dump_feats", False):
            return
        if self._dump_count >= self._dump_max:
            return

        # 反归一化，用于 overlay 到原图（ImageNet 均值方差）
        def denorm(img3chw: torch.Tensor):
            mean = torch.tensor([0.485, 0.456, 0.406], device=img3chw.device).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=img3chw.device).view(3,1,1)
            x = img3chw * std + mean
            return x.clamp(0,1)

        # 颜色映射（优先 matplotlib, fallback HSV）
        _mpl_cmap = None
        try:
            import matplotlib.cm as cm
            _mpl_cmap = cm.get_cmap({'turbo':'turbo','magma':'magma','jet':'jet'}.get(self._dump_colormap,'turbo'))
        except Exception:
            _mpl_cmap = None

        def colorize(gray01: torch.Tensor) -> np.ndarray:
            # gray01: [H,W] in [0,1]
            g = gray01.detach().cpu().numpy()
            g = np.clip(g, 0.0, 1.0)
            if _mpl_cmap is not None:
                rgb = (_mpl_cmap(g)[..., :3] * 255.0).astype(np.uint8)  # [H,W,3]
                return rgb
            else:
                # HSV fallback: H: 240°→0° (blue→red), S=1, V=gray
                h = (1.0 - g) * 240.0  # 0..240
                s = np.ones_like(g)
                v = g
                # HSV→RGB
                c = v * s
                x = c * (1 - np.abs((h / 60.0) % 2 - 1))
                m = v - c
                rgb = np.zeros((*g.shape, 3), dtype=np.float32)
                conds = [
                    (0 <= h) & (h < 60),
                    (60 <= h) & (h < 120),
                    (120 <= h) & (h < 180),
                    (180 <= h) & (h < 240),
                ]
                vals = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c)]
                for cond, (rr, gg, bb) in zip(conds, vals):
                    rgb[cond] = np.stack([rr, gg, bb], axis=-1)[cond]
                rgb = (rgb + m[..., None]) * 255.0
                return rgb.clip(0,255).astype(np.uint8)

        # 纹理增强（Sobel 梯度）
        def sobel_mag(gray: torch.Tensor) -> torch.Tensor:
            # gray: [H,W], float
            kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
            ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
            g = gray[None,None,:,:]
            gx = F.conv2d(g, kx, padding=1)
            gy = F.conv2d(g, ky, padding=1)
            m = torch.sqrt(gx*gx + gy*gy + 1e-12)[0,0]
            return m

        # robust 归一化（百分位裁剪）
        def robust_norm(x: torch.Tensor, pmin=2.0, pmax=98.0):
            qmin = torch.quantile(x.reshape(-1), pmin/100.0)
            qmax = torch.quantile(x.reshape(-1), pmax/100.0)
            y = (x - qmin) / (qmax - qmin + 1e-6)
            return y.clamp(0,1)

        # 叠加到原图
        def overlay_on_img(rgb_color: np.ndarray, base_img_chw: torch.Tensor) -> np.ndarray:
            base = denorm(base_img_chw).permute(1,2,0).detach().cpu().numpy()  # [H,W,3], 0..1
            H, W = base.shape[:2]
            color = rgb_color
            if color.shape[0] != H or color.shape[1] != W:
                from PIL import Image
                color = np.array(Image.fromarray(color).resize((W,H), resample=Image.BILINEAR))
            alpha = np.clip(self._dump_alpha, 0.0, 1.0)
            out = (1-alpha)* (base*255.0) + alpha* color
            return out.clip(0,255).astype(np.uint8)

        import os
        if self._dump_feats:
            os.makedirs(self._dump_dir, exist_ok=True)

        with torch.no_grad():
            for li, f in enumerate(feats):      # f: [B, C, Hf, Wf]
                B = min(f.shape[0], self._dump_per_batch)
                for bi in range(B):
                    feat = f[bi]                # [C,Hf,Wf]
                    C, Hf, Wf = feat.shape

                    if self._dump_style == "pca":
                        X = feat.permute(1,2,0).reshape(Hf*Wf, C)   # (HW, C)
                        X = X - X.mean(dim=0, keepdim=True)
                        # 用 pca_lowrank 更稳
                        U, S, V = torch.pca_lowrank(X, q=3, center=False)
                        Y = X @ V[:, :3]        # (HW, 3)
                        # 逐通道归一化到 [0,1]
                        Y = Y.reshape(Hf, Wf, 3)
                        Yn = []
                        for ch in range(3):
                            Yn.append(robust_norm(Y[..., ch]))
                        rgb = torch.stack(Yn, dim=-1).clamp(0,1)    # [Hf,Wf,3]
                        rgb = (rgb * 255.0).byte().cpu().numpy()
                        # 叠加 or 上采样
                        if imgs is not None and self._dump_upsample == "orig":
                            rgb_out = overlay_on_img(rgb, imgs[bi])
                        else:
                            # 上采样到更清晰的尺寸也行
                            rgb_out = rgb

                    else:
                        # 基础能量图（通道 L2）
                        energy = torch.linalg.vector_norm(feat, dim=0)  # [Hf,Wf]

                        if self._dump_style == "texture":
                            tex = sobel_mag(energy)
                            # 混合：能量 + 纹理
                            a = float(self._dump_texture_blend)
                            m = robust_norm((1-a)*energy + a*tex)
                        else:
                            m = robust_norm(energy)

                        # 上采样目标：原图 or 特征图
                        if imgs is not None and self._dump_upsample == "orig":
                            Hi, Wi = imgs.shape[-2:]
                            m_up = F.interpolate(m[None,None,:,:], size=(Hi,Wi), mode='bilinear', align_corners=False)[0,0]
                        else:
                            m_up = m

                        # 着色 + 叠加
                        color = colorize(m_up)
                        if imgs is not None and self._dump_upsample == "orig":
                            rgb_out = overlay_on_img(color, imgs[bi])
                        else:
                            rgb_out = color

                    # 保存 PNG
                    from PIL import Image
                    Image.fromarray(rgb_out).save(
                        os.path.join(self._dump_dir, f"{self._dump_count:06d}_{tag}_L{li}_B{bi}.png")
                    )

                    # 可选：保存 tensor
                    if self._dump_save_pt:
                        torch.save(feat.detach().cpu(),
                                   os.path.join(self._dump_dir, f"{self._dump_count:06d}_{tag}_L{li}_B{bi}.pt"))

                self._dump_count += 1


    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            print("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    # def forward(self, tensor_list: NestedTensor):
    #     """ """
    #     # (H, W, B, C)
    #     feats = self.encoder(tensor_list.tensors)
    #     feats = self.projector(feats)
    #     # x: [(B, C, H, W)]
    #     out = []
    #     for feat in feats:
    #         m = tensor_list.mask
    #         assert m is not None
    #         mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
    #             0
    #         ]
    #         out.append(NestedTensor(feat, mask))
    #     return out
    def forward(self, tensor_list: NestedTensor):
        """ """
        # (H, W, B, C)
        x_imgs = tensor_list.tensors                    # 原图张量 [B,3,H,W]（标准化后）
        feats = self.encoder(x_imgs)                    # DINOv3中间层特征（List[B,C,Hf,Wf]）

        # 可选导出：raw（DINOv3中间层）
        if hasattr(self, "_maybe_dump"):
            self._maybe_dump(feats, tag="raw", imgs=x_imgs)

        feats = self.projector(feats)                   # 多尺度金字塔特征（List[B,C,Hp,Wp]）

        # 可选导出：proj（projector之后）
        if hasattr(self, "_maybe_dump"):
            self._maybe_dump(feats, tag="proj", imgs=x_imgs)

        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

            
    


    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)

def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
