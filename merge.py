# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import argparse
# import shutil
# from pathlib import Path

# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# def is_image(p: Path) -> bool:
#     return p.is_file() and p.suffix.lower() in IMG_EXTS

# def scan_images(root: Path):
#     root = Path(root)
#     if not root.exists():
#         return []
#     return [p for p in root.rglob("*") if is_image(p)]

# def make_unique_name(dst_dir: Path, name: str) -> str:
#     """If dst_dir/name exists, append _1, _2 ... before extension."""
#     stem = Path(name).stem
#     ext = Path(name).suffix
#     candidate = name
#     i = 1
#     while (dst_dir / candidate).exists():
#         candidate = f"{stem}_{i}{ext}"
#         i += 1
#     return candidate

# def copy_images(src: Path, dst_dir: Path):
#     paths = scan_images(src)
#     copied = 0
#     for idx, p in enumerate(paths, 1):
#         new_name = make_unique_name(dst_dir, p.name)
#         shutil.copy2(p, dst_dir / new_name)
#         copied += 1
#         if idx % 500 == 0:
#             print(f"[{src}] processed {idx} files...")
#     return copied, len(paths)

# def main():
#     ap = argparse.ArgumentParser("Copy images from two folders into one folder (flatten, no prefixes).")
#     ap.add_argument("--src1", required=True, help="Source images folder 1")
#     ap.add_argument("--src2", required=True, help="Source images folder 2")
#     ap.add_argument("--dst",  required=True, help="Destination folder (will be created)")
#     args = ap.parse_args()

#     src1 = Path(args.src1)
#     src2 = Path(args.src2)
#     dst  = Path(args.dst)
#     dst.mkdir(parents=True, exist_ok=True)

#     c1, t1 = copy_images(src1, dst)
#     c2, t2 = copy_images(src2, dst)

#     print("-" * 60)
#     print(f"Source 1: {src1} | found {t1} images | copied {c1}")
#     print(f"Source 2: {src2} | found {t2} images | copied {c2}")
#     print(f"Total copied: {c1 + c2} -> {dst.resolve()}")

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

def is_txt(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".txt"

def scan_txt(root: Path):
    root = Path(root)
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if is_txt(p)]

def make_unique_name(dst_dir: Path, name: str) -> str:
    """If dst_dir/name exists, append _1, _2 ... before extension."""
    stem = Path(name).stem
    ext = Path(name).suffix
    candidate = name
    i = 1
    while (dst_dir / candidate).exists():
        candidate = f"{stem}_{i}{ext}"
        i += 1
    return candidate

def copy_txts(src: Path, dst_dir: Path):
    files = scan_txt(src)
    copied = 0
    for idx, p in enumerate(files, 1):
        new_name = make_unique_name(dst_dir, p.name)
        shutil.copy2(p, dst_dir / new_name)
        copied += 1
        if idx % 500 == 0:
            print(f"[{src}] processed {idx} files...")
    return copied, len(files)

def main():
    ap = argparse.ArgumentParser("Copy all .txt files from two folders into one folder (flatten).")
    ap.add_argument("--src1", required=True, help="Source folder 1")
    ap.add_argument("--src2", required=True, help="Source folder 2")
    ap.add_argument("--dst",  required=True, help="Destination folder (will be created)")
    args = ap.parse_args()

    src1 = Path(args.src1)
    src2 = Path(args.src2)
    dst  = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    c1, t1 = copy_txts(src1, dst)
    c2, t2 = copy_txts(src2, dst)

    print("-" * 60)
    print(f"Source 1: {src1} | found {t1} txt | copied {c1}")
    print(f"Source 2: {src2} | found {t2} txt | copied {c2}")
    print(f"Total copied: {c1 + c2} -> {dst.resolve()}")

if __name__ == "__main__":
    main()
