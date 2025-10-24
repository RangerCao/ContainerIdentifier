import cv2
import os
import glob


def diagnose_image_file(file_path):
    """
    诊断图片文件问题
    """
    print(f"\n🔍 诊断文件: {file_path}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print("❌ 文件不存在")
        return False

    # 检查文件大小
    file_size = os.path.getsize(file_path)
    print(f"📁 文件大小: {file_size} 字节")

    if file_size == 0:
        print("❌ 文件为空")
        return False

    # 检查文件权限
    if not os.access(file_path, os.R_OK):
        print("❌ 没有读取权限")
        return False

    # 尝试用 PIL 读取（备用方案）
    try:
        from PIL import Image
        pil_img = Image.open(file_path)
        print(f"✅ PIL 可以读取，格式: {pil_img.format}, 尺寸: {pil_img.size}")
        pil_img.close()
    except Exception as e:
        print(f"❌ PIL 也无法读取: {e}")

    # 尝试用 OpenCV 读取
    try:
        img = cv2.imread(file_path)
        if img is None:
            print("❌ OpenCV 返回 None")
            return False
        else:
            print(f"✅ OpenCV 读取成功，尺寸: {img.shape}")
            return True
    except Exception as e:
        print(f"❌ OpenCV 读取异常: {e}")
        return False


def mother_fucker_read_fucking_configuration():
    """你的配置文件读取函数"""
    # 临时修改配置路径指向你的数据集
    return {
        "image_folder": r"D:\ContainerIdentifyer\data\images",
        "supported_extensions": [".jpg", ".jpeg", ".png", ".bmp"],
        "max_display_height": 800,
        "auto_resize": True
    }


def view_images_with_diagnosis():
    """
    带诊断功能的图片查看
    """
    config = mother_fucker_read_fucking_configuration()
    image_folder = config["image_folder"]

    print(f"📁 扫描文件夹: {image_folder}")

    # 检查文件夹是否存在
    if not os.path.exists(image_folder):
        print(f"❌ 文件夹不存在: {image_folder}")
        return

    # 获取所有图片文件
    image_files = []
    for ext in config["supported_extensions"]:
        pattern = os.path.join(image_folder, f"*{ext}")
        files = glob.glob(pattern)
        print(f"找到 {len(files)} 个 {ext} 文件")
        image_files.extend(files)

    print(f"📊 总共找到 {len(image_files)} 个图片文件")

    if not image_files:
        print("❌ 没有找到图片文件！")
        return

    # 测试前5个文件
    print("\n🧪 测试前5个文件:")
    for i, img_path in enumerate(image_files[:5]):
        print(f"\n--- 测试文件 {i + 1} ---")
        diagnose_image_file(img_path)

    # 只显示能正常读取的图片
    valid_images = []
    for img_path in image_files:
        if diagnose_image_file(img_path):
            valid_images.append(img_path)

    print(f"\n🎯 可正常显示的图片: {len(valid_images)}/{len(image_files)}")

    if not valid_images:
        print("❌ 没有可以正常显示的图片")
        return

    # 显示可用的图片
    current_index = 0
    while current_index < len(valid_images):
        img_path = valid_images[current_index]
        image = cv2.imread(img_path)

        if image is not None:
            # 调整大小
            height, width = image.shape[:2]
            max_height = config["max_display_height"]
            if height > max_height:
                scale = max_height / height
                new_width = int(width * scale)
                image = cv2.resize(image, (new_width, max_height))

            filename = os.path.basename(img_path)
            cv2.imshow(f'集装箱图片 - {filename} ({current_index + 1}/{len(valid_images)})', image)

            print(f"🎯 显示: {filename}")
            print("⌨️  操作: n-下一张, p-上一张, q-退出")

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):
                current_index += 1
            elif key == ord('p'):
                current_index = max(0, current_index - 1)
        else:
            current_index += 1

        cv2.destroyAllWindows()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    view_images_with_diagnosis()