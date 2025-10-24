import os
import cv2
import random
import glob
import numpy as np


class AnnotationVisualizer:
    def __init__(self, image_folder, label_folder, class_names=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.class_names = class_names or ['damage']  # 默认类别名称

        # 定义颜色（为不同类别分配不同颜色）
        self.colors = [
            (0, 255, 0),  # 绿色
            (255, 0, 0),  # 蓝色
            (0, 0, 255),  # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
        ]

    def parse_yolo_annotation(self, label_path, img_width, img_height):
        """解析YOLO格式标注"""
        boxes = []
        if not os.path.exists(label_path):
            return boxes

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) == 5:
                    class_id = int(data[0])
                    x_center = float(data[1])
                    y_center = float(data[2])
                    width = float(data[3])
                    height = float(data[4])

                    # 转换为绝对坐标
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(0, min(x2, img_width - 1))
                    y2 = max(0, min(y2, img_height - 1))

                    boxes.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(
                            self.class_names) else f'class_{class_id}',
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 1.0  # 标注文件没有置信度，设为1.0
                    })
        return boxes

    def visualize_single_image(self, image_name, save_path=None, display=True):
        """可视化单张图片的标注"""
        image_path = os.path.join(self.image_folder, image_name)
        label_path = os.path.join(self.label_folder, os.path.splitext(image_name)[0] + '.txt')

        print(f"处理图片: {image_name}")
        print(f"图片路径: {image_path}")
        print(f"标注路径: {label_path}")

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图片: {image_path}")
            return None

        img_height, img_width = image.shape[:2]
        print(f"图片尺寸: {img_width} x {img_height}")

        # 读取并解析标注
        boxes = self.parse_yolo_annotation(label_path, img_width, img_height)
        print(f"找到 {len(boxes)} 个标注框")

        # 绘制标注框
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box['bbox']
            class_name = box['class_name']
            class_id = box['class_id']

            # 选择颜色
            color = self.colors[class_id % len(self.colors)]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 绘制类别标签背景
            label = f"{class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

            # 绘制类别文本
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            print(f"  框 {i + 1}: {class_name} [{x1}, {y1}, {x2}, {y2}]")

        # 显示或保存结果
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"✅ 保存结果到: {save_path}")

        if display:
            # 调整显示大小
            display_width = 800
            scale = display_width / img_width
            display_height = int(img_height * scale)
            display_img = cv2.resize(image, (display_width, display_height))

            cv2.imshow(f'标注可视化: {image_name}', display_img)
            print("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image

    def visualize_random_samples(self, num_samples=5, save_dir=None):
        """随机可视化多个样本"""
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(self.image_folder, ext)))
            all_images.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))

        # 只保留文件名
        all_images = [os.path.basename(img) for img in all_images]

        if not all_images:
            print("❌ 没有找到图片文件！")
            return

        print(f"找到 {len(all_images)} 张图片，随机选择 {num_samples} 张进行可视化")

        # 随机选择样本
        random.seed(42)  # 设置随机种子以便复现
        sample_images = random.sample(all_images, min(num_samples, len(all_images)))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, img_name in enumerate(sample_images):
            print(f"\n{'=' * 50}")
            print(f"📸 可视化样本 {i + 1}/{len(sample_images)}: {img_name}")
            print(f"{'=' * 50}")

            if save_dir:
                save_path = os.path.join(save_dir, f"visualized_{img_name}")
                self.visualize_single_image(img_name, save_path=save_path, display=True)
            else:
                self.visualize_single_image(img_name, display=True)

    def batch_visualize_all(self, save_dir=None, max_images=50):
        """批量可视化所有图片（限制数量避免内存问题）"""
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(self.image_folder, ext)))
            all_images.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))

        all_images = [os.path.basename(img) for img in all_images]

        if not all_images:
            print("❌ 没有找到图片文件！")
            return

        # 限制处理数量
        process_images = all_images[:max_images]
        print(f"找到 {len(all_images)} 张图片，处理前 {len(process_images)} 张")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, img_name in enumerate(process_images):
            print(f"\n[{i + 1}/{len(process_images)}] 处理: {img_name}")

            if save_dir:
                save_path = os.path.join(save_dir, f"annotated_{img_name}")
                self.visualize_single_image(img_name, save_path=save_path, display=False)
            else:
                self.visualize_single_image(img_name, display=True)


def main():
    # 配置路径
    image_folder = r"D:\ContainerIdentifyer\data\images"
    label_folder = r"D:\ContainerIdentifyer\data\labels"

    # 检查路径是否存在
    if not os.path.exists(image_folder):
        print(f"❌ 图片文件夹不存在: {image_folder}")
        return
    if not os.path.exists(label_folder):
        print(f"❌ 标注文件夹不存在: {label_folder}")
        return

    # 创建可视化器
    visualizer = AnnotationVisualizer(image_folder, label_folder)

    print("🎯 标注可视化工具")
    print("1. 随机可视化几个样本")
    print("2. 可视化指定图片")
    print("3. 批量可视化所有图片")

    choice = input("请选择模式 (1/2/3): ").strip()

    if choice == "1":
        # 随机可视化5个样本
        num_samples = int(input("要可视化的样本数量 (默认5): ") or "5")
        save_dir = input("保存目录 (留空则不保存): ").strip() or None
        visualizer.visualize_random_samples(num_samples=num_samples, save_dir=save_dir)

    elif choice == "2":
        # 可视化指定图片
        image_name = input("请输入图片文件名: ").strip()
        if not image_name:
            print("❌ 请输入有效的图片文件名")
            return

        save_path = input("保存路径 (留空则不保存): ").strip() or None
        visualizer.visualize_single_image(image_name, save_path=save_path, display=True)

    elif choice == "3":
        # 批量可视化
        max_images = int(input("最大处理数量 (默认50): ") or "50")
        save_dir = input("保存目录 (必须提供): ").strip()
        if not save_dir:
            print("❌ 批量处理必须提供保存目录")
            return
        visualizer.batch_visualize_all(save_dir=save_dir, max_images=max_images)

    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()