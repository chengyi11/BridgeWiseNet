#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subset images+labels by *groups* so patches from the same original (e.g., 'v-10')
stay together.

- Group ID extraction (default): prefix before '_down<digits>' in filename stem.
  Example: 'v-10_down2__1024__0___824' -> group 'v-10'
- Sample by --count (groups) or --ratio (groups), mutually exclusive.
- Copy (default) / move / symlink files.
- Match labels by stem: foo.jpg <-> foo.txt (configurable by --label-ext).

Author: ChatGPT (bridge)
"""

import argparse
import random
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_args():
    ap = argparse.ArgumentParser("Subset images+labels into a new folder (group-aware)")
    ap.add_argument("--images", required=True, help="Path to images dir")
    ap.add_argument("--labels", required=True, help="Path to labels dir")
    ap.add_argument("--out", required=True, help="Output dir (will contain images/ and labels/)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", type=int, help="Number of GROUPS to take")
    g.add_argument("--ratio", type=float, help="Ratio (0<r<=1) of GROUPS to take")
    ap.add_argument("--label-ext", default=".txt", help="Label file extension (default .txt)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--recursive", action="store_true", help="Recursively search images (default off)")

    # group extraction options
    ap.add_argument("--group-regex", default=r"^(.+?)_down\d+",
                    help="Regex with one capturing group for group-id. Default: prefix before '_down<digits>'")
    ap.add_argument("--fallback-split", default="__",
                    help="If regex fails, split stem by this token and take the first part (default '__').")
    ap.add_argument("--drop-group-if-any-missing-label", action="store_true",
                    help="If set, drop the whole group if ANY image in the group has no label. "
                         "Default: keep images that have labels and skip only the missing ones.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="Copy files (default)")
    mode.add_argument("--move", action="store_true", help="Move files")
    mode.add_argument("--symlink", action="store_true", help="Symlink files")
    return ap.parse_args()

def list_images(root: Path, recursive: bool) -> List[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])

def list_labels(root: Path, label_ext: str) -> Dict[str, Path]:
    return {p.stem: p for p in root.rglob(f"*{label_ext}") if p.is_file()}

def extract_group_id(stem: str, regex: re.Pattern, fallback_split: str) -> str:
    m = regex.match(stem)
    if m and m.group(1):
        return m.group(1)
    # fallback: split by token (e.g., '__'), else by first underscore
    if fallback_split and fallback_split in stem:
        return stem.split(fallback_split)[0]
    if "_" in stem:
        return stem.split("_")[0]
    return stem

def transfer(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)

def main():
    args = parse_args()
    img_dir = Path(args.images)
    lab_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_img = out_dir / "images"
    out_lab = out_dir / "labels"
    mode = "copy"
    if args.move:
        mode = "move"
    elif args.symlink:
        mode = "symlink"

    images = list_images(img_dir, args.recursive)
    if not images:
        print(f"[WARN] no images found in {img_dir}")
        return
    labels = list_labels(lab_dir, args.label_ext)
    if not labels:
        print(f"[WARN] no labels found in {lab_dir} with ext {args.label_ext}")
        return

    # group images (and pair with labels)
    grp_regex = re.compile(args.group_regex)
    groups: Dict[str, List[Tuple[Path, Path]]] = {}
    dropped_missing = 0
    for im in images:
        stem = im.stem
        gid = extract_group_id(stem, grp_regex, args.fallback_split)
        lab = labels.get(stem)
        if lab is None:
            dropped_missing += 1
            if args.drop_group_if_any_missing_label:
                # mark group as invalid by setting to None
                groups.setdefault(gid, None)
            else:
                # just skip this image
                pass
            continue
        if gid not in groups:
            groups[gid] = []
        if groups[gid] is not None:  # if not invalidated
            groups[gid].append((im, lab))

    # clean invalid groups if opted
    if args.drop_group_if_any_missing_label:
        for gid in list(groups.keys()):
            if groups[gid] is None:
                del groups[gid]
            elif len(groups[gid]) == 0:
                del groups[gid]
    else:
        # remove empty groups (no valid pairs)
        for gid in [g for g, lst in groups.items() if not lst]:
            del groups[gid]

    if not groups:
        print("[WARN] no valid groups after pairing labels.")
        return

    group_ids = sorted(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)

    # decide how many groups to take
    if args.count is not None:
        k = max(0, min(args.count, len(group_ids)))
    else:
        if not (0 < args.ratio <= 1):
            raise ValueError("--ratio must be in (0,1].")
        k = max(1, int(round(len(group_ids) * args.ratio)))

    chosen = set(group_ids[:k])

    # summary
    total_files = sum(len(groups[gid]) for gid in group_ids)
    chosen_files = sum(len(groups[gid]) for gid in chosen)
    print(f"[INFO] groups total: {len(group_ids)}, files total: {total_files}")
    print(f"[INFO] choose groups: {k}, chosen files: {chosen_files}")
    if dropped_missing:
        print(f"[INFO] skipped images without labels: {dropped_missing}")

    # transfer
    for gid in group_ids[:k]:
        for im, lb in groups[gid]:
            transfer(im, out_img / im.name, mode=mode)
            transfer(lb, out_lab / (im.stem + args.label_ext), mode=mode)

    print("[DONE] subset created.")
    print(f"  images -> {out_img}")
    print(f"  labels -> {out_lab}")

if __name__ == "__main__":
    main()
