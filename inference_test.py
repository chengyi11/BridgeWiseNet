
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
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

""" Demo Inference script for RF-DETR with easy switch between DINOv2 and DINOv3(local repo)."""
import json
import os
import io
import requests
import supervision as sv
from PIL import Image

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

def resolve_encoder(enc_str: str) -> str:
    """Resolve the encoder string to a valid encoder name.
    Args:
        enc_str (str): The encoder string to resolve.

    Returns:
            str: The resolved encoder name.
    Examples:
        resolve_encoder("v2")  # returns "dinov2_windowed_small"
    """
    enc_str = enc_str.strip().lower()
    enc = ENCODER_ALIASES.get(enc_str, enc_str)
    if enc not in VALID_ENCODERS:
        raise ValueError(f"Unknown encoder '{enc_str}'. Valid: {sorted(list(VALID_ENCODERS))}")
    return enc


def load_class_names(annotations_path):
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    # 创建一个从 class_id 到 name 的映射
    # 注意：模型输出的 class_id 通常是从1开始的
    class_map = {category['id']: category['name'] for category in data['categories']}
    return class_map


## If using DINOv3, ensure you have the local repo and weights set up.
def main(encoder:str = "v3", repo_dir: str = "./dinov3_repo", dino_v_weights_path: str = "./dinov3_weights.pth"):
    """Main function to run the inference demo.

    Args:
        encoder (str): The encoder to use, e.g., "v2", "v3", or exact name.
        repo_dir (str): Path to the local DINOv3 repository.
        dino_v_weights_path (str): Path to the DINOv3 weights file.

    Returns:
        None
    
    Examples:
        main(encoder="v2")
        main(encoder="v3", repo_dir="D:/repos/dinov3", dino_v_weights_path="D:/repos/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    """
    encoder = resolve_encoder(encoder)

    if encoder.startswith("dinov3"):
        print("Using DINOv3 encoder:", encoder)
        # Set the environment variables for DINOv3 repo and weights
        os.environ["DINOV3_REPO"] = repo_dir
        os.environ["DINOV3_WEIGHTS"] = dino_v_weights_path
    elif encoder.startswith("dinov2"):
        print("Using DINOv2 encoder:", encoder)

    # Set env *before* importing your package (your Pydantic defaults read env)
    os.environ["RFD_ENCODER"] = encoder


    # Optional: ensure we don't try Hugging Face first (use Hub fallback).
    # If you DO have HF access+token and want to prefer HF, just remove this line.
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)

    from rfdetr import RFDETRMedium, RFDETRBase
    from rfdetr.util.coco_classes import COCO_CLASSES

    # 1. 定义你的模型信息
    my_trained_weights_path = "/home/user/cyshi_lx/GLH-Bridge-Code-main/work-dir/a_new/rfdetr_dinov3_MBDD/run3/checkpoint.pth"  # ⚠️ 替换成你训练好的.pth文件路径
    my_annotations_path = "/home/user/cyshi_lx/GLH-Bridge-Code-main/datasets/SASI_COCO/test/_annotations.coco.json" # ⚠️ 替换成你的训练集标注文件路径
    
    # 2. 加载类别名称
    class_names_map = load_class_names(my_annotations_path)
    my_num_classes = len(class_names_map)

    # 3. 初始化模型，传入你的权重路径和类别数
    print(f"Loading custom model from: {my_trained_weights_path}")
    print(f"Number of classes: {my_num_classes}")

    model = RFDETRMedium(
        pretrain_weights=my_trained_weights_path,
        num_classes=my_num_classes
    )              # uses encoder="dinov3_base" per your config defaults
    model.optimize_for_inference()

    tracker = sv.ByteTrack() # 初始化 ByteTrack

    # 2. 初始化标注器 (用于可视化)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_info = sv.VideoInfo.from_video_path('/data3/lxt/zhongqiyan/test.mp4')
    with sv.VideoSink("output.mp4", video_info) as sink:
        # 3. 逐帧处理
        for frame in sv.get_video_frames_generator('/data3/lxt/zhongqiyan/test.mp4'):
            # 从你的检测模型获取结果
            detections = model.predict(frame, threshold=0.5)
            
            # 4. 核心步骤：用检测结果更新跟踪器
            # 跟踪器会返回带有 tracker_id 的新 Detections 对象
            tracked_detections = tracker.update_with_detections(detections)
            
            # 5. 可视化
            # 创建标签，现在可以包含 tracker_id 了
            labels = [
                f"#{track_id} {class_names_map.get(class_id, 'unknown')}"
                for class_id, track_id 
                in zip(tracked_detections.class_id, tracked_detections.tracker_id)
            ]
            
            annotated_frame = box_annotator.annotate(frame.copy(), detections=tracked_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=tracked_detections, labels=labels)
            
            # 写入输出视频
            sink.write_frame(annotated_frame)

if __name__ == "__main__":
    main(encoder="v3", repo_dir="/data3/lxt/zhongqiyan/code/dinov3", dino_v_weights_path="/data3/lxt/zhongqiyan/code/checkpoint_dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")