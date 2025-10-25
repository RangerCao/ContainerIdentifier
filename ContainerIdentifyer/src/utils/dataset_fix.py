import os
import shutil
import random


def create_proper_validation_set():
    """从训练集创建正确的验证集"""
    base_dir = r"D:\ContainerIdentifyer\data"

    # 路径定义
    train_images_dir = os.path.join(base_dir, "images", "train")
    train_labels_dir = os.path.join(base_dir, "labels", "train")
    val_images_dir = os.path.join(base_dir, "images", "val")
    val_labels_dir = os.path.join(base_dir, "labels", "val")

    # 确保验证集目录存在
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 获取训练集所有有效文件（既有图片又有标签）
    valid_train_files = []
    for image_file in os.listdir(train_images_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_base = os.path.splitext(image_file)[0]
            label_file = image_base + '.txt'
            label_path = os.path.join(train_labels_dir, label_file)

            # 检查标签文件是否存在且不为空
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                valid_train_files.append(image_file)

    print(f"训练集有效文件数量: {len(valid_train_files)}")

    if len(valid_train_files) == 0:
        print("❌ 训练集中没有有效的文件对（图片+标签）")
        return False

    # 随机选择20%作为验证集
    val_ratio = 0.2
    val_count = min(int(len(valid_train_files) * val_ratio), len(valid_train_files))
    val_files = random.sample(valid_train_files, val_count)

    print(f"从训练集划分 {val_count} 个样本到验证集")

    # 移动文件到验证集
    moved_count = 0
    for image_file in val_files:
        image_base = os.path.splitext(image_file)[0]
        label_file = image_base + '.txt'

        # 源路径
        src_image = os.path.join(train_images_dir, image_file)
        src_label = os.path.join(train_labels_dir, label_file)

        # 目标路径
        dst_image = os.path.join(val_images_dir, image_file)
        dst_label = os.path.join(val_labels_dir, label_file)

        # 移动文件
        shutil.move(src_image, dst_image)
        shutil.move(src_label, dst_label)
        moved_count += 1

    # 删除缓存文件
    cache_files = [
        os.path.join(base_dir, "labels", "train.cache"),
        os.path.join(base_dir, "labels", "val.cache")
    ]
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    print(f"✅ 成功创建验证集: {moved_count} 个样本")
    return True

if __name__ == '__main__':
    create_proper_validation_set()