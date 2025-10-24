import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import glob


# 设置中文字体 - 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 或者使用系统字体
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

class YOLOAnalyzer:
    def __init__(self, image_folder, label_folder):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.class_names = self.get_class_names()

    def get_class_names(self):
        """获取类别名称，需要根据你的数据集修改"""
        # 这里需要你提供具体的类别名称
        # 例如: ['dent', 'crack', 'rust', 'breakage']
        return ['damage']  # 暂时用一个通用类别

    def parse_yolo_annotation(self, label_path, img_width, img_height):
        """解析YOLO格式标注"""
        boxes = []
        with open(label_path, 'r') as f:
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

                    boxes.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(
                            self.class_names) else f'class_{class_id}',
                        'bbox': [x1, y1, x2, y2],
                        'relative_bbox': [x_center, y_center, width, height]
                    })
        return boxes

    def visualize_annotation(self, image_name, save_path=None):
        """可视化标注"""
        # 构建文件路径
        image_path = os.path.join(self.image_folder, image_name)
        label_path = os.path.join(self.label_folder, os.path.splitext(image_name)[0] + '.txt')

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None

        img_height, img_width = image.shape[:2]

        # 读取标注
        if os.path.exists(label_path):
            boxes = self.parse_yolo_annotation(label_path, img_width, img_height)
        else:
            print(f"没有找到标注文件: {label_path}")
            boxes = []

        # 绘制边界框
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            class_name = box['class_name']

            # 绘制矩形
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制类别标签
            label = f"{class_name}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示或保存
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"保存可视化结果: {save_path}")
            return image
        else:
            # 调整显示大小
            display_size = (800, 600)
            display_img = cv2.resize(image, display_size)
            cv2.imshow(f"Annotation: {image_name}", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return image

    def get_all_image_files(self):
        """获取所有图片文件"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(self.image_folder, ext)))
            all_images.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))

        # 只返回文件名
        return [os.path.basename(img) for img in all_images]

    def batch_analyze_dataset(self, sample_ratio=0.1, max_samples=1000):
        """批量分析所有样本"""
        print("🚀 开始批量分析数据集...")

        # 获取所有图片文件
        all_images = self.get_all_image_files()
        print(f"📁 找到 {len(all_images)} 张图片")

        if not all_images:
            print("❌ 没有找到图片文件！")
            return

        # 计算需要分析的样本数量
        sample_count = min(int(len(all_images) * sample_ratio), max_samples, len(all_images))
        print(f"🔍 分析 {sample_count} 个样本 (总数: {len(all_images)})")

        # 随机选择样本（如果需要）
        if sample_count < len(all_images):
            import random
            sample_images = random.sample(all_images, sample_count)
        else:
            sample_images = all_images

        # 统计变量
        total_boxes = 0
        class_counts = Counter()
        bbox_sizes = []
        bbox_aspect_ratios = []
        image_sizes = []
        images_with_boxes = 0
        images_without_boxes = 0

        # 进度跟踪
        from tqdm import tqdm

        print("\n📊 分析进度:")
        for i, img_name in enumerate(tqdm(sample_images)):
            image_path = os.path.join(self.image_folder, img_name)
            label_path = os.path.join(self.label_folder, os.path.splitext(img_name)[0] + '.txt')

            # 读取图片尺寸
            image = cv2.imread(image_path)
            if image is not None:
                img_height, img_width = image.shape[:2]
                image_sizes.append((img_width, img_height))

            # 分析标注文件
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            images_with_boxes += 1
                            total_boxes += len(lines)

                            for line in lines:
                                data = line.strip().split()
                                if len(data) == 5:
                                    class_id = int(data[0])
                                    width = float(data[3])
                                    height = float(data[4])

                                    class_counts[class_id] += 1
                                    bbox_sizes.append(width * height)
                                    bbox_aspect_ratios.append(width / height if height > 0 else 0)
                        else:
                            images_without_boxes += 1
                except Exception as e:
                    print(f"\n❌ 读取标注文件错误 {label_path}: {e}")
            else:
                images_without_boxes += 1

        # 输出详细统计结果
        self._print_detailed_statistics(
            total_boxes, class_counts, bbox_sizes, bbox_aspect_ratios,
            image_sizes, images_with_boxes, images_without_boxes, sample_count
        )

        # 生成可视化报告
        self._generate_visual_report(
            bbox_sizes, bbox_aspect_ratios, image_sizes, class_counts
        )

    def _print_detailed_statistics(self, total_boxes, class_counts, bbox_sizes,
                                   bbox_aspect_ratios, image_sizes, images_with_boxes,
                                   images_without_boxes, sample_count):
        """输出详细统计信息"""
        print("\n" + "=" * 60)
        print("📈 数据集详细统计分析报告")
        print("=" * 60)

        print(f"\n📁 样本统计:")
        print(f"  分析样本数量: {sample_count}")
        print(f"  有标注的图片: {images_with_boxes} ({images_with_boxes / sample_count * 100:.1f}%)")
        print(f"  无标注的图片: {images_without_boxes} ({images_without_boxes / sample_count * 100:.1f}%)")
        print(f"  总边界框数量: {total_boxes}")
        print(f"  平均每图边界框: {total_boxes / max(images_with_boxes, 1):.2f}")

        if image_sizes:
            avg_width = np.mean([size[0] for size in image_sizes])
            avg_height = np.mean([size[1] for size in image_sizes])
            print(f"  平均图片尺寸: {avg_width:.0f}×{avg_height:.0f}")

        print(f"\n🎯 类别分布:")
        if class_counts:
            for class_id, count in class_counts.most_common():
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                percentage = count / total_boxes * 100
                print(f"  {class_name} (ID: {class_id}): {count} 个实例 ({percentage:.1f}%)")
        else:
            print("  没有找到标注信息")

        if bbox_sizes:
            print(f"\n📏 边界框尺寸分析:")
            print(f"  平均相对面积: {np.mean(bbox_sizes):.4f}")
            print(f"  面积标准差: {np.std(bbox_sizes):.4f}")
            print(f"  平均宽高比: {np.mean(bbox_aspect_ratios):.2f}")

            # 目标尺寸分布
            small = sum(1 for s in bbox_sizes if s < 0.01)
            medium = sum(1 for s in bbox_sizes if 0.01 <= s < 0.1)
            large = sum(1 for s in bbox_sizes if s >= 0.1)
            total = len(bbox_sizes)

            print(f"  小目标(<0.01): {small} 个 ({small / total * 100:.1f}%)")
            print(f"  中目标(0.01-0.1): {medium} 个 ({medium / total * 100:.1f}%)")
            print(f"  大目标(>=0.1): {large} 个 ({large / total * 100:.1f}%)")

    def _generate_visual_report(self, bbox_sizes, bbox_aspect_ratios, image_sizes, class_counts):
        """生成可视化报告"""
        if not bbox_sizes:
            print("\n❌ 没有足够的数据生成可视化报告")
            return

        print("\n📊 生成可视化报告中...")

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('集装箱破损数据集分析报告', fontsize=16, fontweight='bold')

        # 1. 边界框面积分布
        axes[0, 0].hist(bbox_sizes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('边界框相对面积分布')
        axes[0, 0].set_xlabel('相对面积')
        axes[0, 0].set_ylabel('数量')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 宽高比分布
        axes[0, 1].hist(bbox_aspect_ratios, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('边界框宽高比分布')
        axes[0, 1].set_xlabel('宽高比')
        axes[0, 1].set_ylabel('数量')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 类别分布
        if class_counts:
            classes = [f'Class {cid}' for cid in class_counts.keys()]
            counts = list(class_counts.values())
            axes[1, 0].bar(classes, counts, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('类别分布')
            axes[1, 0].set_ylabel('实例数量')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 图片尺寸分布
        if image_sizes:
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            axes[1, 1].scatter(widths, heights, alpha=0.5, color='purple')
            axes[1, 1].set_title('图片尺寸分布')
            axes[1, 1].set_xlabel('宽度')
            axes[1, 1].set_ylabel('高度')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        report_path = os.path.join(self.image_folder, 'dataset_analysis_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化报告已保存: {report_path}")

        # 显示图表
        plt.show()

    def batch_visualize_samples(self, num_samples=10, save_dir=None):
        """批量可视化样本"""
        all_images = self.get_all_image_files()

        if not all_images:
            print("❌ 没有找到图片文件！")
            return

        # 随机选择样本
        import random
        sample_images = random.sample(all_images, min(num_samples, len(all_images)))

        print(f"\n🖼️  批量可视化 {len(sample_images)} 个样本...")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, img_name in enumerate(sample_images):
            print(f"  [{i + 1}/{len(sample_images)}] 处理: {img_name}")

            if save_dir:
                save_path = os.path.join(save_dir, f"visualized_{img_name}")
                self.visualize_annotation(img_name, save_path)
            else:
                self.visualize_annotation(img_name)


def main():
    # 配置路径
    image_folder = r"D:\ContainerIdentifyer\data\images"
    label_folder = r"D:\ContainerIdentifyer\data\labels"  # 你需要确认标注文件路径

    # 检查路径是否存在
    if not os.path.exists(image_folder):
        print(f"❌ 图片文件夹不存在: {image_folder}")
        return
    if not os.path.exists(label_folder):
        print(f"❌ 标注文件夹不存在: {label_folder}")
        return

    analyzer = YOLOAnalyzer(image_folder, label_folder)

    # 批量分析所有样本（分析10%的样本，最多1000个）
    analyzer.batch_analyze_dataset(sample_ratio=0.1, max_samples=1000)

    # 批量可视化一些样本（可选）
    # analyzer.batch_visualize_samples(num_samples=5, save_dir="./visualized_samples")


if __name__ == "__main__":
    main()