#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 DOTA OBB(4点旋转框) 转为 YOLO HBB(水平框) 标注：
- 输入：DOTA txt（行：x1 y1 x2 y2 x3 y3 x4 y4 class [difficult]），可含头两行：imagesource/gsd
- 输出：YOLO txt（行：class_id cx cy w h，归一化到 [0,1]）
- 做法：以 OBB 的外接矩形作为 HBB，并裁剪到图像边界后再归一化

依赖：Pillow
    pip install pillow
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def load_classes(classes_path: Optional[Path]) -> Dict[str, int]:
    name2id: Dict[str, int] = {}
    if classes_path and classes_path.is_file():
        with classes_path.open("r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        name2id = {n: i for i, n in enumerate(names)}
        print(f"[INFO] 载入类名 {len(name2id)} 个，自 {classes_path}")
    else:
        print("[INFO] 未提供 --classes，将按遇到的类名动态分配 id，并在输出目录写出 classes.txt")
    return name2id


def save_classes(name2id: Dict[str, int], out_dir: Path):
    # 按 id 排序写回
    names = [None] * len(name2id)
    for n, i in name2id.items():
        if 0 <= i < len(names):
            names[i] = n
    # 若有空洞（动态分配一般不会），按 name 排序补齐
    if any(v is None for v in names):
        names = [n for n, _ in sorted(name2id.items(), key=lambda x: x[1])]
    out_path = out_dir / "classes.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for n in names:
            f.write(f"{n}\n")
    print(f"[INFO] 已写出类名清单：{out_path}")


def parse_dota_line(line: str) -> Optional[Tuple[List[float], str, Optional[int]]]:
    """
    返回 (points[8], class_name, difficult)
    - 如果是 header 或空行，返回 None
    - DOTA 一般行：x1 y1 x2 y2 x3 y3 x4 y4 class [difficult]
    """
    s = line.strip()
    if not s:
        return None
    if s.lower().startswith("imagesource:") or s.lower().startswith("gsd:"):
        return None
    parts = s.split()
    if len(parts) < 9:
        return None  # 不合法
    try:
        coords = list(map(float, parts[:8]))
    except ValueError:
        return None
    cls = parts[8]
    diff = None
    if len(parts) >= 10:
        try:
            diff = int(parts[9])
        except ValueError:
            diff = None
    return coords, cls, diff


def obb_to_hbb(coords: List[float]) -> Tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax


def clip_box(xmin, ymin, xmax, ymax, w, h):
    xmin = max(0.0, min(xmin, w - 1.0))
    xmax = max(0.0, min(xmax, w - 1.0))
    ymin = max(0.0, min(ymin, h - 1.0))
    ymax = max(0.0, min(ymax, h - 1.0))
    # 可能出现 xmin>xmax 或 ymin>ymax（完全越界），再修正为无效框
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def to_yolo(xmin, ymin, xmax, ymax, w, h):
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return cx, cy, bw, bh


def find_image(stem: str, images_dir: Path) -> Optional[Path]:
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".tif", ".tiff", ".bmp"):
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    # 再递归搜一次（有的集可能分子文件夹）
    candidates = list(images_dir.rglob(f"{stem}.*"))
    return candidates[0] if candidates else None


def process_one(
    label_path: Path,
    images_dir: Path,
    out_dir: Path,
    name2id: Dict[str, int],
    dynamic_classes: bool,
    skip_ignored: bool,
    skip_difficult: bool,
    decimals: int = 6,
) -> Tuple[int, int, int]:
    stem = label_path.stem
    img_path = find_image(stem, images_dir)
    if img_path is None:
        print(f"[WARN] 找不到对应图像：{stem}.*  跳过")
        return 0, 0, 0

    with Image.open(img_path) as im:
        w, h = im.size

    out_lines: List[str] = []
    total, kept, skipped = 0, 0, 0

    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_dota_line(line)
            if parsed is None:
                continue
            total += 1
            coords, cls_name, diff = parsed

            if skip_ignored and cls_name == "###":
                skipped += 1
                continue
            if skip_difficult and (diff == 1):
                skipped += 1
                continue

            box = obb_to_hbb(coords)
            clipped = clip_box(*box, w, h)
            if clipped is None:
                skipped += 1
                continue
            xmin, ymin, xmax, ymax = clipped
            cx, cy, bw, bh = to_yolo(xmin, ymin, xmax, ymax, w, h)

            # 过滤极小框（可选）
            if bw <= 0 or bh <= 0:
                skipped += 1
                continue

            # 类别映射
            if cls_name not in name2id:
                if not dynamic_classes:
                    print(f"[WARN] 类别 {cls_name} 不在 classes.txt 中，跳过该目标")
                    skipped += 1
                    continue
                name2id[cls_name] = len(name2id)

            cid = name2id[cls_name]
            out_lines.append(
                f"{cid} {round(cx,decimals)} {round(cy,decimals)} {round(bw,decimals)} {round(bh,decimals)}"
            )
            kept += 1

    # 输出 YOLO 标签
    out_path = out_dir / f"{stem}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ln in out_lines:
            f.write(ln + "\n")

    return total, kept, skipped


def main():
    ap = argparse.ArgumentParser("DOTA OBB -> YOLO HBB 转换器")
    ap.add_argument("--images", type=Path, required=True, help="图像根目录")
    ap.add_argument("--dota-labels", type=Path, required=True, help="DOTA OBB 标注目录（txt）")
    ap.add_argument("--out-labels", type=Path, required=True, help="YOLO HBB 输出目录（txt）")
    ap.add_argument("--classes", type=Path, default=None, help="类名文件（每行一个）。若不提供则动态分配并写出 classes.txt")
    ap.add_argument("--skip-ignored", action="store_true", help="跳过类别为 ### 的忽略标注")
    ap.add_argument("--skip-difficult", action="store_true", help="跳过 difficult=1 的目标（如果存在第10列）")
    ap.add_argument("--decimals", type=int, default=6, help="输出小数位")
    args = ap.parse_args()

    images_dir: Path = args.images
    labels_dir: Path = args.dota_labels
    out_dir: Path = args.out_labels
    out_dir.mkdir(parents=True, exist_ok=True)

    name2id = load_classes(args.classes)
    dynamic_classes = args.classes is None

    label_files = sorted([p for p in labels_dir.rglob("*.txt")])
    if not label_files:
        print(f"[ERROR] 在 {labels_dir} 下未找到任何 .txt 标注文件")
        return

    g_total = g_kept = g_skipped = 0
    for i, lp in enumerate(label_files, 1):
        t, k, s = process_one(
            lp,
            images_dir,
            out_dir,
            name2id,
            dynamic_classes,
            skip_ignored=args.skip_ignored,
            skip_difficult=args.skip_difficult,
            decimals=args.decimals,
        )
        g_total += t
        g_kept += k
        g_skipped += s
        if i % 50 == 0:
            print(f"[INFO] 进度 {i}/{len(label_files)}，累计：总{g_total} 保留{k} 跳过{s}")

    print(f"[DONE] 文件数：{len(label_files)} | 总目标：{g_total} | 保留：{g_kept} | 跳过：{g_skipped}")

    if dynamic_classes and name2id:
        save_classes(name2id, out_dir)

    # 额外写出 data.yaml 片段（便于 Ultralytics 使用）
    yaml_path = out_dir / "data_snippet.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(
            "# 这是一个便捷片段，把它合并到你的 data.yaml 中即可\n"
            f"names: {list(sorted(name2id, key=lambda x: name2id[x]))}\n"
        )
    print(f"[INFO] 已写出类名 YAML 片段：{yaml_path}")


if __name__ == "__main__":
    main()
