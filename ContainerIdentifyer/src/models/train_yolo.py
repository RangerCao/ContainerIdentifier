from ultralytics import YOLO
import os
import yaml


def load_dataset_config():
    """读取YOLO数据集配置文件"""
    config_path = r'D:\ContainerIdentifyer\configs\dataset.yaml'

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请先创建数据集配置文件")
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证必要的配置项
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                print(f"❌ 配置文件中缺少必要字段: {key}")
                return None

        print(f"✅ 成功读取数据集配置文件: {config_path}")
        print(f"   数据集路径: {config['path']}")
        print(f"   训练集: {config['train']}")
        print(f"   验证集: {config['val']}")
        print(f"   类别数: {config['nc']}")
        print(f"   类别名称: {config['names']}")

        return config_path

    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return None


def find_local_model(model_size='n'):
    """查找本地模型文件"""
    # 在脚本父文件夹中查找模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    model_name = f'yolov8{model_size}.pt'

    possible_paths = [
        os.path.join(script_dir, model_name),  # 脚本所在文件夹
        os.path.join(parent_dir, model_name),  # 脚本父文件夹
        os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', model_name),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到本地模型: {path}")
            return path

    return None


def load_model_simple(model_size):
    """简化版模型加载"""
    model_name = f'yolov8{model_size}.pt'

    # 首先尝试直接加载（会触发自动下载）
    try:
        model = YOLO(model_name)
        print(f"✅ 成功加载模型: {model_name}")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ GPU可用: {gpu_count} 个设备")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                print(f"  设备 {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return True, '0'
        else:
            print("⚠️  GPU不可用，将使用CPU训练")
            return False, 'cpu'
    except ImportError:
        print("⚠️  无法导入torch，将使用CPU模式")
        return False, 'cpu'


def get_optimal_batch_size(device):
    """根据设备推荐批次大小"""
    if device == 'cpu':
        return 4

    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

            if gpu_memory >= 12:
                return 32
            elif gpu_memory >= 8:
                return 16
            else:
                return 8
    except:
        pass
    return 8


def train_yolo_model():
    """训练YOLOv8模型"""
    print("🚀 开始训练YOLOv8模型...")

    # 1. 读取数据集配置
    config_path = load_dataset_config()
    if config_path is None:
        print("❌ 无法读取数据集配置，训练终止")
        return None

    # 2. 检查GPU
    gpu_available, device = check_gpu_availability()

    # 3. 选择模型大小
    model_size = input("选择YOLO模型大小 (n/s/m/l/x, 默认n): ").strip().lower() or 'n'

    # 4. 加载模型（会自动下载到脚本所在文件夹）
    model = load_model_simple(model_size)
    if model is None:
        print("❌ 无法加载模型，训练终止")
        return None

    # 5. 设置训练参数
    epochs = int(input("训练轮数 (默认50): ") or "50")
    recommended_batch = get_optimal_batch_size(device)
    batch_size = int(input(f"批次大小 (推荐{recommended_batch}, 默认{recommended_batch}): ") or str(recommended_batch))
    img_size = int(input("图片尺寸 (默认640): ") or "640")

    print(f"\n⚙️  训练配置:")
    print(f"  设备: {device}")
    print(f"  轮数: {epochs}")
    print(f"  批次: {batch_size}")
    print(f"  图片尺寸: {img_size}")
    print(f"  数据集配置: {config_path}")

    # 6. 开始训练
    print("\n🎯 开始训练...")
    try:
        results = model.train(
            data=config_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            workers=4,
            patience=10,
            save=True,
            exist_ok=True,
            verbose=True
        )

        print("✅ 训练完成！")
        return results

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return None


def evaluate_model():
    """评估训练好的模型"""
    print("\n📊 模型评估...")

    # 1. 读取数据集配置
    config_path = load_dataset_config()
    if config_path is None:
        print("❌ 无法读取数据集配置，评估终止")
        return None

    # 找到最新训练的模型
    runs_dir = r'D:\ContainerIdentifyer\runs\detect'
    if os.path.exists(runs_dir):
        train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
        if train_dirs:
            latest_train = sorted(train_dirs)[-1]
            model_path = os.path.join(runs_dir, latest_train, 'weights', 'best.pt')

            if os.path.exists(model_path):
                print(f"🔍 找到训练好的模型: {model_path}")
                model = YOLO(model_path)
                metrics = model.val(
                    data=config_path,
                    split='val'
                )
                print("✅ 评估完成！")
                return metrics
            else:
                print("❌ 未找到训练好的模型权重文件")
        else:
            print("❌ 未找到训练记录")
    else:
        print("❌ 未找到训练结果目录")

    return None


def show_config_template():
    """显示配置文件模板"""
    print("\n📝 数据集配置文件模板 (dataset.yaml):")
    print("=" * 50)
    print("""path: D:/ContainerIdentifyer/data  # 数据集根目录
train: images/train    # 训练集图片路径
val: images/val        # 验证集图片路径
test: images/test      # 测试集图片路径（可选）

nc: 3  # 类别数量
names: ['dent', 'hole', 'rusty']  # 类别名称
""")
    print("=" * 50)
    print("请确保配置文件保存在: D:\\ContainerIdentifyer\\configs\\dataset.yaml")


def main():
    """主函数"""
    print("=" * 50)
    print("      YOLOv8 集装箱破损检测训练工具")
    print("=" * 50)

    while True:
        print("\n请选择操作:")
        print("1. 训练新模型")
        print("2. 评估现有模型")
        print("3. 查看配置文件模板")
        print("4. 退出")

        choice = input("请输入选择 (1/2/3/4): ").strip()

        if choice == "1":
            train_yolo_model()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            show_config_template()
        elif choice == "4":
            print("👋 退出程序")
            break
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()