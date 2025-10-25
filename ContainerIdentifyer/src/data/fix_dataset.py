import os
import glob


def fix_class_ids():
    """修复标注文件中的类别ID"""
    label_folder = r"D:\ContainerIdentifyer\data\labels"

    # 获取所有标注文件
    label_files = glob.glob(os.path.join(label_folder, "**", "*.txt"), recursive=True)

    print(f"找到 {len(label_files)} 个标注文件")

    fixed_count = 0
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    # 将类别ID 1和2都映射为0（因为只有一个类别'damage'）
                    if class_id in [1, 2]:
                        parts[0] = '0'
                        fixed_count += 1
                    new_lines.append(' '.join(parts) + '\n')

            # 写回文件
            with open(label_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {e}")

    print(f"修复了 {fixed_count} 个类别ID")


def check_label_files():
    """检查标注文件"""
    train_label_dir = r"D:\ContainerIdentifyer\data\labels\train"
    val_label_dir = r"D:\ContainerIdentifyer\data\labels\val"

    print("检查训练集标注:")
    train_files = glob.glob(os.path.join(train_label_dir, "*.txt"))
    print(f"训练集标注文件: {len(train_files)} 个")

    print("检查验证集标注:")
    val_files = glob.glob(os.path.join(val_label_dir, "*.txt"))
    print(f"验证集标注文件: {len(val_files)} 个")

    # 检查验证集是否有标注内容
    if val_files:
        sample_file = val_files[0]
        with open(sample_file, 'r') as f:
            content = f.read().strip()
        print(f"验证集样本文件内容: '{content}'")


def update_dataset_yaml():
    """更新数据集配置文件"""
    config_content = """
# YOLO数据集配置文件
path: D:/ContainerIdentifyer/data
train: images/train
val: images/val

# 类别数量 - 根据你的实际类别数修改
nc: 1

# 类别名称 - 根据你的实际类别名称修改
names: ['damage']
"""

    config_path = r"D:\ContainerIdentifyer\configs\dataset.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"✅ 更新数据集配置文件: {config_path}")


if __name__ == "__main__":
    print("🔧 修复数据集问题...")

    # 1. 检查当前状态
    check_label_files()

    # 2. 修复类别ID
    fix_class_ids()

    # 3. 更新配置文件
    update_dataset_yaml()

    print("✅ 修复完成！")