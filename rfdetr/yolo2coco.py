import os
import json
import cv2  # pip install opencv-python-headless
import shutil
from tqdm import tqdm

# ===============================================================================
# 1. 配置你的路径和类别信息 (请根据你的实际情况修改)
# ===============================================================================

# 你的YOLO格式数据集的根目录
# yolo_dataset_root = '/home/taotao/paper_object/zhongqiyan/VOC2007'
yolo_dataset_root = '/home/taotao/paper_object/zhongqiyan/VOC2007'


# 转换后要生成的COCO格式数据集的根目录
coco_output_root = './competition_coco'


CLASS_NAMES = [
    'person',      # 对应 class_id 1
    'bike',        # 对应 class_id 2
    'car',         # 对应 class_id 3
    'bus',         # 对应 class_id 4
    'truck'        # 对应 class_id 5
]

# ===============================================================================
# 2. 核心转换逻辑 (通常不需要修改)
# ===============================================================================

def convert_yolo_to_coco(yolo_root, coco_root, class_names):
    """
    将YOLO格式的数据集转换为COCO格式。
    """
    # 创建COCO目标目录结构
    os.makedirs(os.path.join(coco_root, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(coco_root, 'train2017'), exist_ok=True)
    os.makedirs(os.path.join(coco_root, 'val2017'), exist_ok=True)

    # COCO JSON的基本结构
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 创建类别映射
    for i, name in enumerate(class_names):
        coco_format['categories'].append({
            "id": i+1,
            "name": name,
            "supercategory": "object"
        })

    # 处理 'train' 和 'val' 两个数据集
    for dataset_split in ['train', 'val']:
        print(f"--- Processing '{dataset_split}' split ---")
        
        # 因为你的目录名是 train2007, val2007
        yolo_split_name = dataset_split + '2007'
        coco_split_name = dataset_split + '2017'

        yolo_images_dir = os.path.join(yolo_root, 'images', yolo_split_name)
        yolo_labels_dir = os.path.join(yolo_root, 'labels', yolo_split_name)
        coco_images_dir = os.path.join(coco_root, coco_split_name)

        # 初始化当前split的coco json
        split_coco = coco_format.copy()
        split_coco['images'] = []
        split_coco['annotations'] = []
        
        image_id_counter = 0
        annotation_id_counter = 0

        image_files = [f for f in os.listdir(yolo_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_filename in tqdm(image_files, desc=f"Converting {dataset_split} images"):
            # 1. 处理图片
            image_path = os.path.join(yolo_images_dir, image_filename)
            
            # 读取图片获取尺寸
            try:
                img = cv2.imread(image_path)
                img_height, img_width, _ = img.shape
            except Exception as e:
                print(f"Warning: Could not read image {image_path}, skipping. Error: {e}")
                continue
                
            # 复制图片到COCO目录
            shutil.copy(image_path, os.path.join(coco_images_dir, image_filename))

            # 添加图片信息到COCO json
            image_info = {
                "id": image_id_counter,
                "file_name": image_filename,
                "width": img_width,
                "height": img_height
            }
            split_coco['images'].append(image_info)

            # 2. 处理对应的标注文件
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(yolo_labels_dir, label_filename)

            if not os.path.exists(label_path):
                # 如果图片没有对应的标注文件，直接进入下一张
                image_id_counter += 1
                continue

            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id, x_center, y_center, w, h = map(float, parts)
                    class_id = int(class_id)
                    
                    # 坐标转换：从YOLO (归一化, 中心点) 到 COCO (绝对像素, 左上角)
                    abs_w = w * img_width
                    abs_h = h * img_height
                    x_min = (x_center * img_width) - (abs_w / 2)
                    y_min = (y_center * img_height) - (abs_h / 2)

                    annotation_info = {
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": class_id,
                        "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                        "area": round(abs_w * abs_h, 2),
                        "iscrowd": 0
                    }
                    split_coco['annotations'].append(annotation_info)
                    annotation_id_counter += 1

            image_id_counter += 1

        # 将当前split的json数据写入文件
        json_output_path = os.path.join(coco_root, 'annotations', f'instances_{coco_split_name}.json')
        with open(json_output_path, 'w') as f:
            json.dump(split_coco, f, indent=4)
        print(f"Successfully created '{json_output_path}'")

if __name__ == '__main__':
    # 检查输入路径是否存在
    if not os.path.isdir(yolo_dataset_root):
        print(f"Error: YOLO dataset root not found at '{yolo_dataset_root}'")
    else:
        convert_yolo_to_coco(yolo_dataset_root, coco_output_root, CLASS_NAMES)
        print("\nConversion complete!")
        print(f"COCO dataset is ready at: '{coco_output_root}'")