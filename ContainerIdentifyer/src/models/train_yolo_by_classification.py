import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 分类训练YOLO

class ContainerDamageDataset(Dataset):
    """集装箱破损分类数据集"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DamageClassifier:
    def __init__(self, num_classes=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.model = self._create_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.transform = self._get_transforms()

    def _create_model(self, num_classes):
        """创建分类模型"""
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def _get_transforms(self):
        """数据预处理"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return {'train': train_transform, 'val': val_transform}

    def prepare_data(self, image_folder, label_folder):
        """准备分类数据"""
        print("📁 准备分类数据集...")

        # 获取所有图片
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))

        # 根据标注文件判断是否有破损
        image_paths = []
        labels = []

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            label_file = os.path.join(label_folder, os.path.splitext(img_name)[0] + '.txt')

            # 如果有标注文件，说明有破损（类别1），否则无破损（类别0）
            if os.path.exists(label_file):
                image_paths.append(img_path)
                labels.append(1)  # 有破损
            else:
                image_paths.append(img_path)
                labels.append(0)  # 无破损

        print(f"数据集统计:")
        print(f"  总图片数: {len(image_paths)}")
        print(f"  无破损图片: {labels.count(0)}")
        print(f"  有破损图片: {labels.count(1)}")

        return image_paths, labels

    def train(self, train_loader, val_loader, epochs=10):
        """训练模型"""
        print("🚀 开始训练分类模型...")

        best_accuracy = 0
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  训练损失: {train_loss / len(train_loader):.4f}')
            print(f'  验证损失: {val_loss / len(val_loader):.4f}')
            print(f'  验证准确率: {accuracy:.4f}')

            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), 'best_classifier.pth')
                print(f'  ✅ 保存最佳模型，准确率: {accuracy:.4f}')


def main():
    """主函数"""
    print("=" * 50)
    print("     集装箱破损分类模型训练")
    print("=" * 50)

    # 配置路径
    image_folder = r"D:\ContainerIdentifyer\data\images"
    label_folder = r"D:\ContainerIdentifyer\data\labels"

    # 创建分类器
    classifier = DamageClassifier(num_classes=2)

    # 准备数据
    image_paths, labels = classifier.prepare_data(image_folder, label_folder)

    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 创建数据加载器
    train_dataset = ContainerDamageDataset(train_paths, train_labels, classifier.transform['train'])
    val_dataset = ContainerDamageDataset(val_paths, val_labels, classifier.transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 开始训练
    classifier.train(train_loader, val_loader, epochs=10)


if __name__ == "__main__":
    main()