# create_dataset_split.py
import os
import random
import shutil
from sklearn.model_selection import train_test_split

def create_classification_dataset():
    """创建二分类数据集（有破损 vs 无破损）"""
    # 根据标注文件判断哪些图片有破损
    pass