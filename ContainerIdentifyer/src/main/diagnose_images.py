import os
import cv2
import glob
from PIL import Image


def check_permissions_and_files():
    """
    检查权限和文件可访问性
    """
    # 多个可能的路径
    possible_paths = [
        r"D:\ContainerIdentifyer\src\resources\数据集3713\images\train",
        r"C:\Users\27138\Downloads\2025年第六届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2025年MathorCup大数据挑战赛-赛道A初赛\数据集3713\images\train"
    ]

    for base_path in possible_paths:
        print(f"\n🔍 检查路径: {base_path}")

        # 检查路径是否存在
        if not os.path.exists(base_path):
            print("   ❌ 路径不存在")
            continue

        # 检查权限
        if not os.access(base_path, os.R_OK):
            print("   ❌ 没有读取权限")
            continue

        print("   ✅ 路径存在且有读取权限")

        # 列出文件夹内容
        try:
            items = os.listdir(base_path)
            print(f"   📁 文件夹内容: {len(items)} 个项目")

            # 查找图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []

            for item in items[:10]:  # 只显示前10个
                item_path = os.path.join(base_path, item)
                if os.path.isfile(item_path):
                    _, ext = os.path.splitext(item)
                    if ext.lower() in image_extensions:
                        image_files.append(item_path)
                        print(f"      📄 {item}")

            if image_files:
                print(f"   🖼️  找到 {len(image_files)} 个图片文件")
                # 测试第一个图片文件
                test_image_path = image_files[0]
                test_single_image(test_image_path)
                break
            else:
                print("   ❌ 没有找到图片文件")

        except PermissionError as e:
            print(f"   ❌ 权限错误: {e}")
        except Exception as e:
            print(f"   ❌ 其他错误: {e}")


def test_single_image(image_path):
    """
    测试单个图片文件的读取
    """
    print(f"\n🧪 测试图片: {os.path.basename(image_path)}")

    # 检查文件权限
    if not os.access(image_path, os.R_OK):
        print("   ❌ 文件没有读取权限")
        return False

    # 检查文件大小
    file_size = os.path.getsize(image_path)
    print(f"   📏 文件大小: {file_size} 字节")

    if file_size == 0:
        print("   ❌ 文件为空")
        return False

    # 方法1: 使用OpenCV读取
    print("   1. 尝试OpenCV读取...")
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            print(f"      ✅ OpenCV成功 - 尺寸: {img_cv.shape}")
        else:
            print("      ❌ OpenCV返回None")
    except Exception as e:
        print(f"      ❌ OpenCV错误: {e}")

    # 方法2: 使用PIL读取
    print("   2. 尝试PIL读取...")
    try:
        img_pil = Image.open(image_path)
        print(f"      ✅ PIL成功 - 格式: {img_pil.format}, 尺寸: {img_pil.size}")
        img_pil.close()
        return True
    except Exception as e:
        print(f"      ❌ PIL错误: {e}")
        return False


def create_simple_test():
    """
    创建一个简单的测试，使用绝对路径直接测试
    """
    print("\n🎯 直接路径测试:")

    # 直接指定一个具体的图片文件路径
    test_files = [
        r"D:\ContainerIdentifyer\src\resources\数据集3713\images\train\1323.jpg",
        r"C:\Users\27138\Downloads\2025年第六届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2025年MathorCup大数据挑战赛-赛道A初赛\数据集3713\images\train\1323.jpg"
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"📁 找到文件: {test_file}")
            test_single_image(test_file)
        else:
            print(f"❌ 文件不存在: {test_file}")


def copy_images_to_safe_location():
    """
    将图片复制到没有中文路径的安全位置
    """
    safe_folder = r"D:\ContainerIdentifyer\data\images"

    # 创建安全文件夹
    os.makedirs(safe_folder, exist_ok=True)
    print(f"\n📁 创建安全文件夹: {safe_folder}")

    # 找到源图片文件夹
    source_folders = [
        r"D:\ContainerIdentifyer\src\resources\数据集3713\images\train",
        r"C:\Users\27138\Downloads\2025年第六届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2025年MathorCup大数据挑战赛-赛道A初赛\数据集3713\images\train"
    ]

    for source_folder in source_folders:
        if os.path.exists(source_folder):
            print(f"🔍 从 {source_folder} 复制图片...")

            # 复制前5个图片进行测试
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            copied_count = 0

            for ext in image_extensions:
                pattern = os.path.join(source_folder, f"*{ext}")
                for img_path in glob.glob(pattern)[:2]:  # 每个扩展名复制2个
                    try:
                        import shutil
                        filename = os.path.basename(img_path)
                        dest_path = os.path.join(safe_folder, filename)
                        shutil.copy2(img_path, dest_path)
                        print(f"   ✅ 复制: {filename}")
                        copied_count += 1
                    except Exception as e:
                        print(f"   ❌ 复制失败 {filename}: {e}")

            if copied_count > 0:
                print(f"🎉 成功复制 {copied_count} 个图片到安全位置")
                return safe_folder

    return None


if __name__ == "__main__":
    print("=" * 60)
    print("           集装箱图片诊断和修复工具")
    print("=" * 60)

    # 1. 检查权限和文件
    check_permissions_and_files()

    # 2. 直接测试
    create_simple_test()

    # 3. 复制到安全位置（推荐）
    print("\n🔄 尝试复制图片到安全位置...")
    safe_folder = copy_images_to_safe_location()

    if safe_folder:
        print(f"\n🎉 现在可以使用安全路径: {safe_folder}")
        print("   修改你的配置文件，使用这个路径:")
        print(f'   "image_folder": "{safe_folder}"')
    else:
        print("\n❌ 无法复制图片，请检查源文件夹权限")