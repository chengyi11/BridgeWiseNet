import json
import os

# ===============================================================================
# 1. 配置你要修正的JSON文件路径
# ===============================================================================
# 将你的 competition_coco 数据集路径放在这里
# 这个脚本会自动寻找其中的 train/_annotations.coco.json 和 valid/_annotations.coco.json
DATASET_ROOT = './competition_coco_roboflow'

# ===============================================================================
# 2. 核心修正逻辑 (通常不需要修改)
# ===============================================================================

def fix_coco_json_ids(json_path):
    """
    读取一个COCO JSON文件，将其中的所有类别ID和标注ID减1，并覆盖保存。
    """
    if not os.path.exists(json_path):
        print(f"错误：文件不存在于 '{json_path}'")
        return

    print(f"正在处理文件: {json_path} ...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 修正 "categories" 列表中的 "id"
    # 检查 'categories' 键是否存在并且是一个列表
    if 'categories' in data and isinstance(data['categories'], list):
        for category in data['categories']:
            if 'id' in category:
                category['id'] -= 1
    else:
        print(f"警告：在 {json_path} 中没有找到 'categories' 列表。")


    # 2. 修正 "annotations" 列表中的 "category_id"
    # 检查 'annotations' 键是否存在并且是一个列表
    if 'annotations' in data and isinstance(data['annotations'], list):
        for annotation in data['annotations']:
            if 'category_id' in annotation:
                annotation['category_id'] -= 1
    else:
        print(f"警告：在 {json_path} 中没有找到 'annotations' 列表。")


    # 3. 将修改后的内容写回原文件
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"文件处理完成，ID已更新！")


if __name__ == '__main__':
    # 构建需要处理的文件路径列表
    train_json = os.path.join(DATASET_ROOT, 'train', '_annotations.coco.json')
    valid_json = os.path.join(DATASET_ROOT, 'valid', '_annotations.coco.json')
    
    files_to_fix = [train_json, valid_json]
    
    for file_path in files_to_fix:
        fix_coco_json_ids(file_path)
    
    print("\n所有JSON文件ID修正完毕！")