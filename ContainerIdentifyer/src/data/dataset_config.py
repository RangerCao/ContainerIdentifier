# 数据集配置 - 需要你根据实际情况修改
DATASET_CONFIG = {
    "name": "container_damage",
    "image_dir": r"D:\ContainerIdentifyer\data\images",
    "label_dir": r"D:\ContainerIdentifyer\data\labels",  # 需要确认
    "class_names": [
        "dent",        # 凹陷
        "crack",       # 裂缝
        "rust",        # 锈蚀
        "breakage",    # 破损
        "scratch"      # 划痕
    ],
    "input_size": 640,  # YOLO常用尺寸
    "train_ratio": 0.8,
    "val_ratio": 0.2
}