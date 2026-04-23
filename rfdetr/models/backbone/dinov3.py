# ------------------------------------------------------------------------
# Note: This file is an original wrapper that *loads* DINOv3 models.
# Using DINOv3 code/weights is subject to that license’s restrictions.
# DINOv3 itself is licensed under the DINOv3 License; see https://github.com/facebookresearch/dinov3/tree/main?tab=License-1-ov-file
# ------------------------------------------------------------------------

from typing import Sequence, Optional, List, Dict, Any
import os
import torch
import torch.nn as nn

from rfdetr.models.adapters.wss import WSSAdapter2D, WSSConfig

# HF ids for ViT (/16)
_HF_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base":  "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
# channels per size (for projector wiring)
_SIZE2WIDTH = {"small": 384, "base": 768, "large": 1024}
# torch.hub entrypoints in the official repo
_HUB_ENTRY = {"small": "dinov3_vits16", "base": "dinov3_vitb16", "large": "dinov3_vitl16"}

class DinoV3(nn.Module):
    """
    RF-DETR-facing DINOv3 wrapper:
      - forward(x) -> List[B, C, H/16, W/16] (one per selected layer)
      - _out_feature_channels: List[int]
      - export(): no-op (kept for parity)
    """

    def __init__(
        self,
        shape: Sequence[int] = (640, 640),
        out_feature_indexes: Sequence[int] = (2, 5, 8, 11),
        size: str = "base",
        patch_size: int = 16,
        # new knobs:
        load_dinov3_weights: bool = True,
        hf_token: Optional[str] = None,
        repo_dir: Optional[str] = None,     # path to local dinov3 clone
        weights: Optional[str] = None,      # path or URL to *.pth (hub)
        pretrained_name: Optional[str] = None,
        # WSS adapter (Wavelet-Selective Shrinkage)
        wss_enable: bool = False,
        wss_layers: Optional[Sequence[int]] = None,
        wss_levels: int = 1,
        wss_reduction_dim: int = 64,
        **__,
        ):
        """
        A DINOv3 wrapper for RF-DETR.

        Args:
            shape (Sequence[int]): Input image shape (H, W).
            out_feature_indexes (Sequence[int]): Layer indexes to return.
            size (str): DINOv3 model size: "small", "base", or "large".
            patch_size (int): Patch size for the model.
            load_dinov3_weights (bool): If True, load DINOv3 weights from HF or hub.
            hf_token (Optional[str]): Hugging Face token for private models.
            repo_dir (Optional[str]): Path to the local DINOv3 repository.
            weights (Optional[str]): Path to the DINOv3 weights file.
            pretrained_name (Optional[str]): Pretrained model name for HF.
        """
        
        super().__init__()
        assert size in _HF_IDS, f"size must be one of {list(_HF_IDS)}, got {size}"

        self.shape = tuple(shape)
        self.patch_size = int(patch_size)
        self.model_patch = 16
        self.num_register_tokens = 0
        self.hidden_size = _SIZE2WIDTH[size]
        self._out_feature_channels = [self.hidden_size] * len(out_feature_indexes)
        self.out_feature_indexes = list(out_feature_indexes)

        # WSS config
        self.wss_enable = bool(wss_enable)
        self.wss_layers = list(wss_layers) if wss_layers is not None else [2, 5, 8, 11]
        self._wss_cfg = WSSConfig(levels=int(wss_levels), reduction_dim=int(wss_reduction_dim))
        self._wss_hooks: List[Any] = []
        self.wss_adapters = nn.ModuleDict()  # filled after encoder is ready

        # Allow env overrides (so you don't have to touch code)
        repo_dir = repo_dir or os.getenv("DINOV3_REPO")
        weights  = weights  or os.getenv("DINOV3_WEIGHTS")
        hub_id   = pretrained_name or _HF_IDS[size]

        # 1) Try HF if weights are allowed and a token is available
        use_hf = load_dinov3_weights and (hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN"))
        if use_hf:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(
                hub_id,
                token=True if hf_token is None else hf_token,
                output_hidden_states=True,
                return_dict=True,
            )
            # pick config info when available
            cfg = self.encoder.config
            self.hidden_size = int(getattr(cfg, "hidden_size", self.hidden_size))
            self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
            self.model_patch = int(getattr(cfg, "patch_size", self.model_patch))
        else:
            # 2) Fallback to PyTorch Hub (local repo + weights path or URL)
            if not (repo_dir and weights):
                raise RuntimeError(
                    "HF access unavailable/gated. Set DINOV3_REPO and DINOV3_WEIGHTS (or pass repo_dir/weights) "
                    "to load via torch.hub as per the DINOv3 README."
                )
            entry = _HUB_ENTRY[size]
            self.encoder = torch.hub.load(repo_dir, entry, source="local", weights=weights)
            # best-effort introspection (these attrs may or may not exist on hub module)
            self.num_register_tokens = int(getattr(self.encoder, "num_register_tokens", 0))
            self.model_patch = int(getattr(self.encoder, "patch_size", self.model_patch))

        # initialize WSS adapters & register hooks (if enabled)
        if self.wss_enable:
            self._init_wss_and_hooks()

    def freeze_base(self):
        """Freeze backbone parameters but keep adapters trainable."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # keep WSS trainable
        for p in self.wss_adapters.parameters():
            p.requires_grad_(True)

    # ------------------------ WSS integration ------------------------

    def _find_vit_blocks(self) -> Optional[nn.ModuleList]:
        """Best-effort discovery of the transformer block list.

        We search for the *largest* nn.ModuleList whose elements look like
        transformer blocks. This works across HF and hub variants.
        """
        candidates: List[nn.ModuleList] = []
        for m in self.encoder.modules():
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                # heuristics: homogeneous type, and has parameters
                types = {type(x) for x in m}
                if len(types) == 1 and any(p.requires_grad is not None for p in m[0].parameters()):
                    candidates.append(m)
        if not candidates:
            return None
        # pick the longest module list
        candidates.sort(key=lambda x: len(x), reverse=True)
        return candidates[0]

    def _init_wss_and_hooks(self):
        blocks = self._find_vit_blocks()
        if blocks is None:
            # Can't safely inject "between blocks"; we will still expose wrappers,
            # but no hooks will be registered.
            return

        num_blocks = len(blocks)
        # normalize negative indices
        norm_layers = []
        for idx in self.wss_layers:
            li = idx if idx >= 0 else num_blocks + idx
            if 0 <= li < num_blocks:
                norm_layers.append(li)
        self.wss_layers = sorted(set(norm_layers))

        # one adapter per selected block index
        # hidden_size might differ after loading config
        for li in self.wss_layers:
            self.wss_adapters[str(li)] = WSSAdapter2D(self.hidden_size, cfg=self._wss_cfg)

        self._last_hw: Optional[tuple[int, int]] = None

        def _hook_fn(block_idx: int):
            def hook(module, inputs, output):
                if self._last_hw is None:
                    return output
                H, W = self._last_hw
                hp, wp = H // self.model_patch, W // self.model_patch
                expected_hw = hp * wp

                def _apply(hidden: torch.Tensor) -> torch.Tensor:
                    if hidden.dim() != 3:
                        return hidden
                    B, S, C = hidden.shape
                    if expected_hw <= 0 or expected_hw > S:
                        return hidden
                    patch_start = S - expected_hw
                    patch = hidden[:, patch_start:, :]
                    patch_map = patch.view(B, hp, wp, C).permute(0, 3, 1, 2).contiguous()
                    patch_map = self.wss_adapters[str(block_idx)](patch_map)
                    patch_new = patch_map.permute(0, 2, 3, 1).contiguous().view(B, expected_hw, C)
                    if patch_start == 0:
                        return patch_new
                    return torch.cat([hidden[:, :patch_start, :], patch_new], dim=1)

                if torch.is_tensor(output):
                    return _apply(output)
                if isinstance(output, (list, tuple)) and len(output) > 0 and torch.is_tensor(output[0]):
                    out0 = _apply(output[0])
                    if isinstance(output, tuple):
                        return (out0,) + tuple(output[1:])
                    else:
                        return [out0] + list(output[1:])
                return output

            return hook

        # register hooks
        for li in self.wss_layers:
            h = blocks[li].register_forward_hook(_hook_fn(li))
            self._wss_hooks.append(h)

    def export(self):  # parity with dinov2 path
        pass

    def _tokens_to_map(self, hidden_state: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        """
        Accepts either:
        - [B, 1+R+HW, C]  (CLS + register + patch tokens)
        - [B, HW, C]      (no special tokens)
        Returns:
        - [B, C, H/ps, W/ps]
        """
        ps = self.model_patch
        assert H % ps == 0 and W % ps == 0, f"Input must be divisible by patch size {ps}, got {(H, W)}"
        hp, wp = H // ps, W // ps
        C = hidden_state.shape[-1]

        if hidden_state.dim() == 2:
            # e.g., [HW, C] (no batch) -> try to recover batch
            seq = hidden_state.shape[0]
            assert seq % B == 0, f"Cannot infer batch from tokens of shape {hidden_state.shape} with B={B}"
            hidden_state = hidden_state.view(B, seq // B, C)

        assert hidden_state.dim() == 3, f"Expected tokens [B, S, C], got {tuple(hidden_state.shape)}"
        S = hidden_state.shape[1]
        expected_hw = hp * wp

        if S == expected_hw:
            seq = hidden_state  # already patch tokens
        elif S >= expected_hw + 1 + self.num_register_tokens:
            # drop CLS + registers, then take the last expected_hw tokens
            seq = hidden_state[:, 1 + self.num_register_tokens :, :]
            seq = seq[:, -expected_hw:, :]
        else:
            # unknown extra tokens count; take the last expected_hw tokens
            seq = hidden_state[:, -expected_hw:, :]

        return seq.view(B, hp, wp, C).permute(0, 3, 1, 2).contiguous()
        
    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape

        if self.wss_enable:
            # cache for hooks
            self._last_hw = (H, W)

        # --- HF path (AutoModel) -----------------------------
        # Heuristic: HF models have .config and accept pixel_values=...
        if hasattr(self.encoder, "config"):
            out = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states  # tuple/list: embeddings + each layer
            feats = [self._tokens_to_map(hs[i], B, H, W) for i in self.out_feature_indexes]
            return feats

        # --- Hub path: try get_intermediate_layers -----------
        if hasattr(self.encoder, "get_intermediate_layers"):
            try:
                max_idx = max(self.out_feature_indexes)
                hs_list = self.encoder.get_intermediate_layers(x, n=max_idx + 1, reshape=False)
                proc = []
                for h in hs_list:
                    # some impls return (tokens, cls) tuples
                    if isinstance(h, (list, tuple)):
                        h = h[0]
                    proc.append(h)
                feats = [self._tokens_to_map(proc[i], B, H, W) for i in self.out_feature_indexes]
                return feats
            except Exception:
                pass  # fall through to plain forward

        # --- Hub path: try forward_features ------------------
        if hasattr(self.encoder, "forward_features"):
            try:
                ff = self.encoder.forward_features(x)
                # Common patterns: dict with patch tokens or tensors
                if isinstance(ff, dict):
                    cand = (
                        ff.get("x_norm_patchtokens", None)
                        or ff.get("patch_tokens", None)
                        or ff.get("tokens", None)
                        or next((t for t in ff.values() if torch.is_tensor(t) and t.dim() >= 2), None)
                    )
                else:
                    cand = ff
                if cand is not None:
                    if torch.is_tensor(cand) and cand.dim() == 4:
                        # Already a spatial map [B, C, Hp, Wp]
                        # Repeat to match requested out_feature_indexes count
                        C = cand.shape[1]
                        if C != self.hidden_size:
                            self.hidden_size = int(C)
                            self._out_feature_channels = [self.hidden_size] * len(self.out_feature_indexes)
                        return [cand for _ in self.out_feature_indexes]
                    # Otherwise assume tokens
                    tokens = cand
                    # If [HW, C] or [B*HW, C], _tokens_to_map will handle reshape
                    feats = [self._tokens_to_map(tokens, B, H, W) for _ in self.out_feature_indexes]
                    return feats
            except Exception:
                pass  # fall through

        # --- Plain forward fallback --------------------------
        out = self.encoder(x)
        # Normalize to a tensor
        if isinstance(out, (list, tuple)):
            # pick first tensor-like item
            out = next((t for t in out if torch.is_tensor(t)), out[0])
        if not torch.is_tensor(out):
            raise RuntimeError(
                "DINOv3 hub module returned an unsupported output type for tracing. "
                "Prefer the HF path or a hub build exposing intermediate layers."
            )

        # Case A: already spatial map [B, C, Hp, Wp]
        if out.dim() == 4:
            C = out.shape[1]
            if C != self.hidden_size:
                self.hidden_size = int(C)
                self._out_feature_channels = [self.hidden_size] * len(self.out_feature_indexes)
            return [out for _ in self.out_feature_indexes]

        # Case B/C: tokens [B, S, C] or [S, C] (batchless)
        tokens = out
        if tokens.dim() == 2:
            # Let _tokens_to_map do the reshaping using B
            feats = [self._tokens_to_map(tokens, B, H, W) for _ in self.out_feature_indexes]
            return feats
        elif tokens.dim() == 3:
            feats = [self._tokens_to_map(tokens, B, H, W) for _ in self.out_feature_indexes]
            return feats

        raise RuntimeError(f"Unsupported hub output shape: {tuple(out.shape)}")
