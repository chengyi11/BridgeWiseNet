#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan COCO dataset for corrupted/missing images and write a cleaned annotations file.
Usage:
  python sanitize_coco.py \
    --images-root /path/to/images/root \
    --ann /path/to/annotations.json \
    --out-ann /path/to/annotations.clean.json \
    --bad-dir /path/to/bad_images   # optional: move bad files here
"""

import argparse, json, os, shutil
from PIL import Image

def is_image_ok(path: str) -> bool:
    if (not os.path.exists(path)) or os.path.isdir(path):
        return False
    try:
        # 0 字节或极小文件直接认为坏
        if os.path.getsize(path) < 64:
            return False
    except Exception:
        return False
    try:
        with Image.open(path) as im:
            im.verify()  # 只做一致性校验，不解码像素
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, help="COCO images root (the base dir of file_name paths)")
    ap.add_argument("--ann", required=True, help="Original COCO annotations JSON")
    ap.add_argument("--out-ann", required=True, help="Output cleaned annotations JSON")
    ap.add_argument("--bad-dir", default=None, help="If set, move bad images here")
    args = ap.parse_args()

    with open(args.ann, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_by_fname = {}
    for img in data["images"]:
        id_by_fname[img["file_name"]] = img["id"]

    good_ids, bad_ids, bad_files = set(), set(), []

    total = len(data["images"])
    for i, img in enumerate(data["images"], 1):
        rel = img["file_name"]
        abspath = os.path.join(args.images_root, rel)
        ok = is_image_ok(abspath)
        if ok:
            good_ids.add(img["id"])
        else:
            bad_ids.add(img["id"])
            bad_files.append(abspath)
        if i % 1000 == 0 or i == total:
            print(f"[{i}/{total}] checked... good={len(good_ids)} bad={len(bad_ids)}", flush=True)

    if args.bad_dir:
        os.makedirs(args.bad_dir, exist_ok=True)
        for p in bad_files:
            try:
                if os.path.exists(p):
                    dst = os.path.join(args.bad_dir, os.path.basename(p))
                    if os.path.exists(dst):
                        # 防重名
                        base, ext = os.path.splitext(dst)
                        k = 1
                        while os.path.exists(f"{base}_{k}{ext}"):
                            k += 1
                        dst = f"{base}_{k}{ext}"
                    shutil.move(p, dst)
            except Exception as e:
                print(f"[WARN] move failed: {p} -> {e}")

    cleaned_images = [img for img in data["images"] if img["id"] in good_ids]
    cleaned_annotations = [ann for ann in data["annotations"] if ann["image_id"] in good_ids]

    cleaned = {
        "images": cleaned_images,
        "annotations": cleaned_annotations,
        "categories": data.get("categories", []),
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
    }

    with open(args.out_ann, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False)

    print("==== Summary ====")
    print(f"Total images: {total}")
    print(f"Good images : {len(cleaned_images)}")
    print(f"Bad images  : {len(bad_ids)}")
    print(f"Kept anns   : {len(cleaned_annotations)}  (from {len(data['annotations'])})")
    print(f"Wrote       : {args.out_ann}")

if __name__ == "__main__":
    main()
