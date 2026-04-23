# # ------------------------------------------------------------------------
# # RF-DETR
# # Copyright ...
# # ------------------------------------------------------------------------

# """ Single-image Inference demo for RF-DETR with easy switch between DINOv2 and DINOv3. """
# import json
# import os
# import io
# import requests
# import numpy as np
# import supervision as sv
# from PIL import Image

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

# def resolve_encoder(enc_str: str) -> str:
#     enc_str = enc_str.strip().lower()
#     enc = ENCODER_ALIASES.get(enc_str, enc_str)
#     if enc not in VALID_ENCODERS:
#         raise ValueError(f"Unknown encoder '{enc_str}'. Valid: {sorted(list(VALID_ENCODERS))}")
#     return enc

# def load_class_names(annotations_path):
#     with open(annotations_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     # COCO: id -> name（很多数据集 id 从1开始）
#     return {cat['id']: cat['name'] for cat in data.get('categories', [])}

# def load_image_any(path_or_url: str) -> Image.Image:
#     """支持本地路径或 http(s) URL。"""
#     if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
#         resp = requests.get(path_or_url, timeout=15)
#         resp.raise_for_status()
#         return Image.open(io.BytesIO(resp.content)).convert("RGB")
#     return Image.open(path_or_url).convert("RGB")

# def main(
#     encoder: str = "v3",
#     repo_dir: str = "/home/user/cyshi_lx/dinov3-main",
#     dino_v_weights_path: str = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/dinov3-vitl16-sat493/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
#     image_path: str = "/home/user/cyshi_lx/GLH-Bridge-Code-main/datasets/SASI/images/test/J50F012022DOM_x10752_y0.png",     # ⚠️ 这里改成你的图片路径或URL
#     save_path: str = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/test",
#     threshold: float = 0.5,
# ):
#     encoder = resolve_encoder(encoder)

#     # DINOv3 的本地仓库/权重（如果用 v3）
#     if encoder.startswith("dinov3"):
#         print("Using DINOv3 encoder:", encoder)
#         os.environ["DINOV3_REPO"] = repo_dir
#         os.environ["DINOV3_WEIGHTS"] = dino_v_weights_path
#     elif encoder.startswith("dinov2"):
#         print("Using DINOv2 encoder:", encoder)

#     # 让底层默认用本地/torch.hub，不强制走HF（如需HF可以去掉这行）
#     os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
#     os.environ["RFD_ENCODER"] = encoder  # 你的封装里读取这个

#     from rfdetr import RFDETRMedium, RFDETRBase  # 你环境里已有这些封装
#     # （可选）如果你想从 COCO json 里读类名，填路径；否则置为 None
#     my_annotations_path =  "/path/to/_annotations.coco.json" # ⚠️ 修改成你的
#     class_names_map = load_class_names(my_annotations_path) if os.path.isfile(my_annotations_path) else None
#     my_trained_weights_path = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/rfdetr_dinov3_MBDD/run3/checkpoint.pth"  # ⚠️ 修改成你的

#     # 若用 COCO json 统计类别数（注意：这里是你训练集的类别数，不含背景）
#     if class_names_map is not None:
#         my_num_classes = len(class_names_map)
#     else:
#         # 如果没有 json，可手填
#         my_num_classes = 1  # ⚠️ 改成你的类别数

#     print(f"Loading custom model from: {my_trained_weights_path}")
#     print(f"Number of classes: {my_num_classes}")

#     model = RFDETRMedium(
#         pretrain_weights=my_trained_weights_path,
#         num_classes=my_num_classes
#     )
#     # model.optimize_for_inference()

#     # 读取图片 → numpy (H,W,3) BGR/RGB 对 supervision 都OK，这里用RGB
#     img = load_image_any(image_path)
#     frame = np.array(img)  # RGB uint8

#     # 推理（你封装的 predict 支持 numpy/PIL）
#     detections: sv.Detections = model.predict(frame, threshold=threshold)

#     # 生成标签文本
#     if class_names_map is not None:
#         # 注意：很多实现里的 labels 是 0..N-1；COCO 的 category id 常从 1 开始
#         labels_txt = []
#         for cid, conf in zip(detections.class_id, detections.confidence):
#             name = class_names_map.get(int(cid), class_names_map.get(int(cid) + 1, str(int(cid))))
#             labels_txt.append(f"{name} {conf:.2f}")
#     else:
#         labels_txt = [f"{int(cid)} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]

#     # 可视化并保存
#     box_annotator = sv.BoxAnnotator()
#     label_annotator = sv.LabelAnnotator()

#     annotated = box_annotator.annotate(frame.copy(), detections=detections)
#     annotated = label_annotator.annotate(annotated, detections=detections, labels=labels_txt)

#     # 保存
#     Image.fromarray(annotated).save(save_path)
#     print(f"Saved: {save_path}  (n={len(detections)})")

# if __name__ == "__main__":
#     main(
#         encoder="dinov3_large",
#         repo_dir="/home/user/cyshi_lx/dinov3-main",
#         dino_v_weights_path="/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/dinov3-vitl16-sat493/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
#         image_path="/home/user/cyshi_lx/GLH-Bridge-Code-main/datasets/GLH-Bridge/train_crop_down2/train/t-13_down2__1024__1648___5376.png",   # ← 输入图片
#         save_path="/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/test/out.jpg",
#         threshold=0.5,



        
#     )



# ------------------------------------------------------------------------
# RF-DETR
# Copyright ...
# ------------------------------------------------------------------------

""" Single-image OR Folder Inference demo for RF-DETR (DINOv2 / DINOv3). """
import json
import os
import io
import requests
import numpy as np
import supervision as sv
from PIL import Image
from pathlib import Path

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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def resolve_encoder(enc_str: str) -> str:
    enc_str = enc_str.strip().lower()
    enc = ENCODER_ALIASES.get(enc_str, enc_str)
    if enc not in VALID_ENCODERS:
        raise ValueError(f"Unknown encoder '{enc_str}'. Valid: {sorted(list(VALID_ENCODERS))}")
    return enc

def load_class_names(annotations_path):
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # COCO: id -> name（很多数据集 id 从1开始）
    return {cat['id']: cat['name'] for cat in data.get('categories', [])}

def load_image_any(path_or_url: str) -> Image.Image:
    """支持本地路径或 http(s) URL。"""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=15)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")

def ensure_jpg_path(out_path: Path) -> Path:
    # 若无扩展名则补 .jpg；若有其它扩展名也可以直接用 .jpg 统一输出
    if out_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return out_path.with_suffix(".jpg")
    return out_path

def annotate_and_save(frame_rgb: np.ndarray,
                      detections: sv.Detections,
                      class_names_map: dict | None,
                      out_path: Path,
                      threshold: float):
    # 文本标签
    if class_names_map is not None:
        labels_txt = []
        for cid, conf in zip(detections.class_id, detections.confidence):
            # cid 常为 0..N-1；COCO id 常从 1 开始 → 兼容两种
            name = class_names_map.get(int(cid), class_names_map.get(int(cid) + 1, str(int(cid))))
            labels_txt.append(f"{name} {conf:.2f}")
    else:
        labels_txt = [f"{int(cid)} {conf:.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]

    # 可视化
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(frame_rgb.copy(), detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections, labels=labels_txt)

    # 保存（保证有扩展名）
    out_path = ensure_jpg_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(str(out_path), format="JPEG")
    print(f"[Saved] {out_path}  (n={len(detections)})")

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def iter_images(root: Path):
    # 递归遍历
    if root.is_file() and is_image_file(root):
        yield root
    elif root.is_dir():
        for p in root.rglob("*"):
            if is_image_file(p):
                yield p

def main(
    encoder: str = "v3",
    repo_dir: str = "/home/user/cyshi_lx/dinov3-main",
    dino_v_weights_path: str = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/dinov3-vitl16-sat493/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    image_path: str = "/home/user/cyshi_lx/GLH-Bridge-Code-main/datasets/SASI/images/test/J50F012022DOM_x10752_y0.png",  # 文件或目录
    save_path: str | None = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/test/out.jpg",  # 单图时生效
    save_dir: str | None = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/test_batch",     # 目录时生效
    threshold: float = 0.5,
):
    encoder = resolve_encoder(encoder)

    # DINOv3 的本地仓库/权重（如果用 v3）
    if encoder.startswith("dinov3"):
        print("Using DINOv3 encoder:", encoder)
        os.environ["DINOV3_REPO"] = repo_dir
        os.environ["DINOV3_WEIGHTS"] = dino_v_weights_path
    elif encoder.startswith("dinov2"):
        print("Using DINOv2 encoder:", encoder)

    # 不强制走 HF
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    os.environ["RFD_ENCODER"] = encoder

    from rfdetr import RFDETRMedium, RFDETRBase  # 你环境里已有这些封装

    # （可选）COCO 类名
    my_annotations_path = "/path/to/_annotations.coco.json"  # 若没有就设为 None
    if my_annotations_path and os.path.isfile(my_annotations_path):
        class_names_map = load_class_names(my_annotations_path)
        my_num_classes = len(class_names_map)
    else:
        class_names_map = None
        my_num_classes = 1  # ⚠️ 改成你训练时的类别数（不含背景）

    my_trained_weights_path = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/rfdetr_dinov3_MBDD/run3/checkpoint.pth"

    print(f"Loading custom model from: {my_trained_weights_path}")
    print(f"Number of classes: {my_num_classes}")

    model = RFDETRMedium(
        pretrain_weights=my_trained_weights_path,
        num_classes=my_num_classes
    )
    # 如需更快但会影响中间特征导出，请再开启
    # model.optimize_for_inference()

    input_path = Path(image_path)
    if input_path.is_file():
        # --- 单图 ---
        img = load_image_any(str(input_path))
        frame = np.array(img)  # RGB
        dets: sv.Detections = model.predict(frame, threshold=threshold)

        out_p = Path(save_path) if save_path else input_path.with_suffix(".out.jpg")
        annotate_and_save(frame, dets, class_names_map, out_p, threshold)
    elif input_path.is_dir():
        # --- 文件夹批处理（递归）---
        out_root = Path(save_dir or "./preds")
        total, ok = 0, 0
        for p in iter_images(input_path):
            total += 1
            try:
                img = load_image_any(str(p))
                frame = np.array(img)
                dets: sv.Detections = model.predict(frame, threshold=threshold)

                # 镜像原相对路径到输出目录
                rel = p.relative_to(input_path)
                out_p = out_root / rel
                out_p = ensure_jpg_path(out_p)
                annotate_and_save(frame, dets, class_names_map, out_p, threshold)
                ok += 1
            except Exception as e:
                print(f"[Skip] {p}  due to error: {e}")
        print(f"Done. {ok}/{total} images processed. Output: {out_root}")
    else:
        raise FileNotFoundError(f"Not found: {input_path}")

if __name__ == "__main__":
    main(
        encoder="dinov3_large",
        repo_dir="/home/user/cyshi_lx/dinov3-main",
        dino_v_weights_path="/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/dinov3-vitl16-sat493/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        image_path="/home/user/cyshi_lx/GLH-Bridge-Code-main/datasets/partSASI/train/images/",  # ← 目录也可以
        save_path="/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/test/out.jpg",                   # 单图时用
        save_dir="/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/test_batch",                      # 目录时用
        threshold=0.5,
    )
