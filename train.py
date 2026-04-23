# # train_single.py
# import argparse
# import os
# import platform


# ENCODER_ALIASES = {
#     "dinov2": "dinov2_windowed_small",
#     "v2": "dinov2_windowed_small",
#     "dinov2_small": "dinov2_windowed_small",
#     "dinov2_base": "dinov2_windowed_base",
#     "dinov3": "dinov3_base",
#     "v3": "dinov3_base",
# }

# VALID_ENCODERS = {
#     "dinov2_windowed_small",
#     "dinov2_windowed_base",
#     "dinov3_small",
#     "dinov3_base",
#     "dinov3_large",
# }

# def parse_args():
#     ap = argparse.ArgumentParser("RF-DETR Medium single-run trainer (v2 or v3)")
#     ap.add_argument("--data", default="/data4_ssd/lxt/Datasets/zhongqiyanData/coco_roboflow/", help="Dataset root with train/valid/test 指定 COCO 格式数据集所在的根目录")
#     ap.add_argument("--out", default="./runs", help="Root output dir for TB & checkpoints 指定训练结果（日志、模型权重）的保存目录")
#     ap.add_argument("--epochs", type=int, default=50)
#     ap.add_argument("--resolution", type=int, default=1280)
#     ap.add_argument("--bs", type=int, default=8, help="Batch size per iteration") ##TODO #Not actually applying, not worked out why
#     ap.add_argument("--workers", type=int, default=8,
#                     help="DataLoader workers (default: 0 on Windows, else 2)")
#     ap.add_argument("--encoder", default="dinov3_base",
#                     help=("dinov2|v2|dinov2_small|dinov2_base|dinov3|v3|"
#                           "dinov3_small|dinov3_base|dinov3_large or exact name"))
#     ap.add_argument("--name", default="Test0829_1280", help="Optional run name (subdir under --out) 为本次训练指定一个自定义的名字，方便在 ./runs 目录下区分。")

#     # Optional local DINOv3 assets
#     ap.add_argument("--dinov3-repo", default="/data3/lxt/zhongqiyan/code/dinov3", help="Local DINOv3 repo (sets DINOV3_REPO)")
#     ap.add_argument("--dinov3-weights", default="/data3/lxt/zhongqiyan/code/checkpoint_dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", help="Path to DINOv3 .pth (sets DINOV3_WEIGHTS)")
#     return ap.parse_args()

# def resolve_encoder(enc_str: str) -> str:
#     enc_str = enc_str.strip().lower()
#     enc = ENCODER_ALIASES.get(enc_str, enc_str)
#     if enc not in VALID_ENCODERS:
#         raise ValueError(f"Unknown encoder '{enc_str}'. Valid: {sorted(list(VALID_ENCODERS))}")
#     return enc

# def main():
#     args = parse_args()

#     # Safer default on Windows for DataLoader workers
#     if args.workers is None:
#         import platform
#         args.workers = 0 if platform.system() == "Windows" else 2

#     encoder_name = resolve_encoder(args.encoder)

#     # Set env *before* importing your package (your Pydantic defaults read env)\
#     # 通过 os.environ 将 encoder_name 设置为环境变量
#     os.environ["RFD_ENCODER"] = encoder_name
#     if args.dinov3_repo:
#         os.environ["DINOV3_REPO"] = args.dinov3_repo
#     if args.dinov3_weights:
#         os.environ["DINOV3_WEIGHTS"] = args.dinov3_weights

#     # Now import project code
#     from rfdetr import RFDETRBase, RFDETRMedium
#     from rfdetr.config import RFDETRBaseConfig, RFDETRMediumConfig  # to override config cleanly

#     # Two thin wrappers so we can control encoder and pretrain at construction time
#     # 作者没有直接使用 RFDETRMedium 这个类，而是为 DINOv2 和 DINOv3 分别创建了一个“薄包装”子类。
#     class RFDETRBaseV2(RFDETRMedium):
#         def get_model_config(self, **kwargs):
#             # keep RF-DETR pretrain (default) for v2
#             return RFDETRMediumConfig(encoder="dinov2_windowed_small", **kwargs)

#     class RFDETRBaseV3(RFDETRMedium):
#         def get_model_config(self, **kwargs):
#             # IMPORTANT: disable RF-DETR pretrain for v3 to avoid shape mismatches
#             return RFDETRMediumConfig(encoder="dinov3_base", pretrain_weights=None, **kwargs)

#     # Output dir (separate subdirs so TB shows two runs side-by-side)
#     if args.name:
#         run_name = args.name
#     else:
#         tag = "DINOv2" if encoder_name.startswith("dinov2") else "DINOv3"
#         run_name = f"{tag}_Base"
#     out_dir = os.path.join(args.out, run_name)
#     os.makedirs(out_dir, exist_ok=True)

#     # Build and train
#     ModelCls = RFDETRBaseV3 if encoder_name.startswith("dinov3") else RFDETRBaseV2
#     model = ModelCls()

#     print(f"\n=== Training RF-DETR Base with encoder: {encoder_name} ===")
#     print(f"Dataset: {args.data}")
#     print(f"Output : {out_dir}")

#     train_kwargs = dict(
#         dataset_dir=args.data,
#         output_dir=out_dir,
#         epochs=args.epochs,
#         batch_size=args.bs,  #TODO #Not actually applying, not worked out why
#         num_workers=args.workers,
#         tensorboard=True,
#         run_test=True,
#         resolution=args.resolution
#     )
#     # NOTE: train() expects kwargs, not a TrainConfig instance
#     model.train(**train_kwargs)

#     print("\nDone. View in TensorBoard with:")
#     print(f"  tensorboard --logdir {args.out}")
#     print("Open http://127.0.0.1:6006")

# if __name__ == "__main__":
#     main()



# train_single.py
import argparse
import os
import platform
from typing import Optional, List, Tuple

ENCODER_ALIASES = {
    "dinov2": "dinov2_windowed_small",
    "v2": "dinov2_windowed_small",
    "dinov2_small": "dinov2_windowed_small",
    "dinov2_base": "dinov2_windowed_base",
    "dinov3": "dinov3_base",
    "v3": "dinov3_base",
}

VALID_ENCODERS = {
    "dinov2_windowed_small",
    "dinov2_windowed_base",
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
}

def parse_args():
    ap = argparse.ArgumentParser("RF-DETR Medium single-run trainer (v2 or v3)")
    ap.add_argument("--data", default="/data4_ssd/lxt/Datasets/zhongqiyanData/coco_roboflow/", help="Dataset root with train/valid/test 指定 COCO 格式数据集所在的根目录")
    ap.add_argument("--out", default="./runs", help="Root output dir for TB & checkpoints 指定训练结果（日志、模型权重）的保存目录")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--bs", type=int, default=4, help="Batch size per iteration")  # TODO: Not actually applying
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers (default: 0 on Windows, else 2)")
    ap.add_argument("--encoder", default="dinov3_large",
                    help=("dinov2|v2|dinov2_small|dinov2_base|dinov3|v3|"
                          "dinov3_small|dinov3_base|dinov3_large or exact name"))
    ap.add_argument("--name", default="Test0829_1280", help="Optional run name (subdir under --out) 为本次训练指定一个自定义的名字，方便在 ./runs 目录下区分。")

    # Optional local DINOv3 assets
    ap.add_argument("--dinov3-repo", default="/data3/lxt/zhongqiyan/code/dinov3", help="Local DINOv3 repo (sets DINOV3_REPO)")
    ap.add_argument("--dinov3-weights", default="/data3/lxt/zhongqiyan/code/checkpoint_dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", help="Path to DINOv3 .pth (sets DINOV3_WEIGHTS)")

    # ==== Resume options ====
    ap.add_argument("--resume", default=None,
                    help="断点续训：指定 checkpoint 路径，或用 'auto'/'latest' 自动搜最新，'best' 优先搜 best。")
    ap.add_argument("--resume-strict", action="store_true",
                    help="若开启，--resume=auto/latest/best 找不到 checkpoint 时直接报错退出。")

    return ap.parse_args()

def resolve_encoder(enc_str: str) -> str:
    enc_str = enc_str.strip().lower()
    enc = ENCODER_ALIASES.get(enc_str, enc_str)
    if enc not in VALID_ENCODERS:
        raise ValueError(f"Unknown encoder '{enc_str}'. Valid: {sorted(list(VALID_ENCODERS))}")
    return enc

def _list_ckpts(search_dir: str, max_depth: int = 2) -> List[Tuple[str, float]]:
    """
    在 search_dir 中递归（有限深度）搜集 .pth/.pt 文件，返回 (path, mtime) 列表
    """
    results = []
    if not os.path.isdir(search_dir):
        return results
    for root, dirs, files in os.walk(search_dir):
        depth = os.path.relpath(root, search_dir).count(os.sep)
        if depth > max_depth:
            # 限制递归深度，避免工作目录太大
            dirs[:] = []
            continue
        for f in files:
            if f.endswith(".pth") or f.endswith(".pt"):
                p = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(p)
                except OSError:
                    continue
                results.append((p, mtime))
    return results

def _pick_best(cands: List[Tuple[str, float]], mode: str) -> Optional[str]:
    """
    从候选中挑选：
    - mode == 'best'：优先包含 'best' 的文件名，其次 'final'，否则退化为 latest（按 mtime 最大）
    - mode == 'latest'：mtime 最大的
    """
    if not cands:
        return None
    names = [p for p, _ in cands]
    if mode == "best":
        # 先找包含 best 的
        best_like = [t for t in cands if "best" in os.path.basename(t[0]).lower()]
        if best_like:
            return sorted(best_like, key=lambda x: x[1], reverse=True)[0][0]
        # 再找 final
        final_like = [t for t in cands if "final" in os.path.basename(t[0]).lower()]
        if final_like:
            return sorted(final_like, key=lambda x: x[1], reverse=True)[0][0]
        # 否则退化 latest
    # latest
    return sorted(cands, key=lambda x: x[1], reverse=True)[0][0]

def _auto_resume_ckpt(out_dir: str, mode: str = "latest") -> Optional[str]:
    """
    在 out_dir 及常见子目录里自动搜 checkpoint。
    """
    search_roots = [
        out_dir,
        os.path.join(out_dir, "checkpoints"),
        os.path.join(out_dir, "weights"),
        os.path.join(out_dir, "ckpts"),
    ]
    all_cands: List[Tuple[str, float]] = []
    for d in search_roots:
        all_cands.extend(_list_ckpts(d, max_depth=2))
    # 常见命名优先：best / latest / last
    # 但最终用 _pick_best 控制
    return _pick_best(all_cands, "best" if mode == "best" else "latest")

def main():
    args = parse_args()

    # Safer default on Windows for DataLoader workers
    if args.workers is None:
        args.workers = 0 if platform.system() == "Windows" else 2

    encoder_name = resolve_encoder(args.encoder)

    # Set env *before* importing your package (your Pydantic defaults read env)
    os.environ["RFD_ENCODER"] = encoder_name
    if args.dinov3_repo:
        os.environ["DINOV3_REPO"] = args.dinov3_repo
    if args.dinov3_weights:
        os.environ["DINOV3_WEIGHTS"] = args.dinov3_weights

    # Now import project code
    from rfdetr import RFDETRBase, RFDETRMedium
    from rfdetr.config import RFDETRBaseConfig, RFDETRMediumConfig  # to override config cleanly

    # Two thin wrappers so we can control encoder and pretrain at construction time
    class RFDETRBaseV2(RFDETRMedium):
        def get_model_config(self, **kwargs):
            # keep RF-DETR pretrain (default) for v2
            return RFDETRMediumConfig(encoder="dinov2_windowed_small", **kwargs)

    class RFDETRBaseV3(RFDETRMedium):
        def get_model_config(self, **kwargs):
            # IMPORTANT: disable RF-DETR pretrain for v3 to avoid shape mismatches
            return RFDETRMediumConfig(encoder="dinov3_large", pretrain_weights=None, **kwargs)

    # Output dir (subdir so TB shows two runs side-by-side)
    if args.name:
        run_name = args.name
    else:
        tag = "DINOv2" if encoder_name.startswith("dinov2") else "DINOv3"
        run_name = f"{tag}_Base"
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ==== Resolve resume path ====
    resume_path: Optional[str] = None
    if args.resume:
        req = str(args.resume).strip().lower()
        if req in ("auto", "latest", "best"):
            mode = "best" if req == "best" else "latest"
            resume_path = _auto_resume_ckpt(out_dir, mode=mode)
            if resume_path:
                print(f"[resume] auto mode='{mode}'  →  {resume_path}")
            elif args.resume_strict:
                raise FileNotFoundError(f"[resume] auto mode='{mode}' 未在目录 '{out_dir}' 下找到可续训的 checkpoint")
            else:
                print(f"[resume] 未找到可用的 checkpoint（mode='{mode}'），将从头训练。")
        else:
            # user-provided path
            if os.path.isfile(args.resume):
                resume_path = args.resume
                print(f"[resume] 使用显式指定的 checkpoint: {resume_path}")
            elif args.resume_strict:
                raise FileNotFoundError(f"[resume] 指定的 checkpoint 文件不存在: {args.resume}")
            else:
                print(f"[resume] 指定的 checkpoint 文件不存在: {args.resume}，将从头训练。")

    # 如果找到了，就同步到环境变量（某些实现可能读 env）
    if resume_path:
        os.environ["RFD_RESUME"] = resume_path

    # Build model
    ModelCls = RFDETRBaseV3 if encoder_name.startswith("dinov3") else RFDETRBaseV2
    model = ModelCls(wss_enable=True, prism_enable=False)

    print(f"\n=== Training RF-DETR Base with encoder: {encoder_name} ===")
    print(f"Dataset: {args.data}")
    print(f"Output : {out_dir}")
    if resume_path:
        print(f"Resume : {resume_path}")

    train_kwargs = dict(
        dataset_dir=args.data,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.bs,   # TODO: Not actually applying
        num_workers=args.workers,
        tensorboard=True,
        run_test=True,
        resolution=args.resolution
    )
    # 将 resume 传给训练（若底层不识别，会被忽略；但我们同时设置了 env 作为兜底）
    if resume_path:
        train_kwargs["resume"] = resume_path

    # NOTE: train() expects kwargs, not a TrainConfig instance
    model.train(**train_kwargs)

    print("\nDone. View in TensorBoard with:")
    print(f"  tensorboard --logdir {args.out}")
    print("Open http://127.0.0.1:6006")

if __name__ == "__main__":
    main()
