import os
import json
import cv2  # pip install opencv-python-headless
import shutil
from tqdm import tqdm

# ===============================================================================
# 1. 配置你的路径和类别信息 (请根据你的实际情况修改)
# ===============================================================================

# 你的YOLO格式数据集的根目录
yolo_dataset_root = '/data4_ssd/lxt/Datasets/zhongqiyanData/VOC2007/'

# 转换后要生成的COCO格式数据集的根目录
coco_output_root = '/data4_ssd/lxt/Datasets/zhongqiyanData//coco_roboflow' # 保持不变

# 你的6个类别名称，顺序与YOLO的class_id (0-5) 严格对应
CLASS_NAMES = [
    'person',           # 对应 YOLO class_id 1
    'bike',             # 对应 YOLO class_id 2
    'car',              # 对应 YOLO class_id 3
    'bus',              # 对应 YOLO class_id 4
    'truck'             # 对应 YOLO class_id 5
]

# ===============================================================================
# 3. 核心转换逻辑 (V3 - Roboflow-Compliant)
# ===============================================================================

def convert_yolo_to_coco_roboflow(yolo_root, coco_root, class_names):
    """
    将YOLO格式的数据集转换为代码期望的 "Roboflow COCO" 格式。
    - 图片和标注JSON都放在各自的 split 文件夹下。
    - 标注文件名为 _annotations.coco.json
    """
    # 先删除旧的目录，避免文件混淆
    if os.path.exists(coco_root):
        print(f"Removing existing directory: {coco_root}")
        shutil.rmtree(coco_root)

    # COCO JSON的基本结构
    coco_format = {
        "info": {}, "licenses": [], "images": [], "annotations": [], "categories": []
    }

    # 创建类别映射 (Roboflow的id通常从0开始，但COCO标准工具从1开始，我们仍坚持从1开始更稳妥)
    for i, name in enumerate(class_names):
        coco_format['categories'].append({
            "id": i,
            "name": name,
            "supercategory": "object"
        })
    
    # 映射 'train' 和 'val' 到 'train' 和 'valid'
    split_mapping = {'train': 'train', 'val': 'valid'}

    for yolo_split, coco_split in split_mapping.items():
        print(f"--- Processing '{coco_split}' split ---")
        
        # 创建Roboflow格式的目标目录
        coco_split_dir = os.path.join(coco_root, coco_split)
        os.makedirs(coco_split_dir, exist_ok=True)

        yolo_images_dir = os.path.join(yolo_root, 'images', yolo_split + '2007')
        yolo_labels_dir = os.path.join(yolo_root, 'labels', yolo_split + '2007')

        split_coco = coco_format.copy()
        split_coco['images'] = []
        split_coco['annotations'] = []
        
        image_id_counter = 0
        annotation_id_counter = 0

        image_files = [f for f in os.listdir(yolo_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_filename in tqdm(image_files, desc=f"Converting {coco_split} images"):
            image_path = os.path.join(yolo_images_dir, image_filename)
            
            try:
                img = cv2.imread(image_path)
                img_height, img_width, _ = img.shape
            except Exception as e:
                print(f"Warning: Could not read image {image_path}, skipping. Error: {e}")
                continue
                
            # 复制图片到新的 split 目录中
            shutil.copy(image_path, os.path.join(coco_split_dir, image_filename))

            image_info = {
                "id": image_id_counter, "file_name": image_filename,
                "width": img_width, "height": img_height
            }
            split_coco['images'].append(image_info)

            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(yolo_labels_dir, label_filename)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) != 5: continue
                        
                        class_id, x_center, y_center, w, h = map(float, parts)
                        class_id = int(class_id) - 1
                        
                        abs_w = w * img_width
                        abs_h = h * img_height
                        x_min = (x_center * img_width) - (abs_w / 2)
                        y_min = (y_center * img_height) - (abs_h / 2)

                        annotation_info = {
                            "id": annotation_id_counter, "image_id": image_id_counter,
                            "category_id": class_id,
                            "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                            "area": round(abs_w * abs_h, 2),
                            "iscrowd": 0
                        }
                        split_coco['annotations'].append(annotation_info)
                        annotation_id_counter += 1
            image_id_counter += 1

        # 将json数据写入到 split 目录中，并命名为 _annotations.coco.json
        json_output_path = os.path.join(coco_split_dir, '_annotations.coco.json')
        with open(json_output_path, 'w') as f:
            json.dump(split_coco, f, indent=4)
        print(f"Successfully created '{json_output_path}'")

if __name__ == '__main__':
    if not os.path.isdir(yolo_dataset_root):
        print(f"Error: YOLO dataset root not found at '{yolo_dataset_root}'")
    else:
        convert_yolo_to_coco_roboflow(yolo_dataset_root, coco_output_root, CLASS_NAMES)
        print("\nConversion complete!")
        print(f"Roboflow-style COCO dataset is ready at: '{coco_output_root}'")