import torch
import subprocess
import sys


def check_gpu_detailed():
    """详细检查GPU状态"""
    print("=" * 50)
    print("          GPU 诊断工具")
    print("=" * 50)

    # 1. 检查PyTorch版本和CUDA支持
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ PyTorch检测不到CUDA，可能的原因:")
        print("   - 安装了CPU版本的PyTorch")
        print("   - NVIDIA驱动未安装")
        print("   - CUDA工具包未安装")
        return False

    # 2. 检查GPU数量和信息
    gpu_count = torch.cuda.device_count()
    print(f"检测到的GPU数量: {gpu_count}")

    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")

        # 测试GPU计算
        try:
            x = torch.randn(3, 3).cuda(i)
            y = torch.randn(3, 3).cuda(i)
            z = x + y
            print(f"  计算测试: ✅ 正常")
        except Exception as e:
            print(f"  计算测试: ❌ 失败 - {e}")

    # 3. 检查NVIDIA驱动
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        print("\n✅ NVIDIA驱动正常")
    except:
        print("\n❌ 未找到nvidia-smi，可能驱动未安装")

    # 4. 检查环境变量
    print(f"\n环境变量:")
    print(f"  CUDA_VISIBLE_DEVICES: {torch.cuda.get_device_name(0) if gpu_count > 0 else '未设置'}")

    return gpu_count > 0


def check_pytorch_installation():
    """检查PyTorch安装版本"""
    print("\n" + "=" * 50)
    print("        PyTorch安装信息")
    print("=" * 50)

    # 检查是否是CUDA版本
    cuda_version = torch.version.cuda
    if cuda_version:
        print(f"✅ PyTorch CUDA版本: {cuda_version}")
    else:
        print("❌ PyTorch CPU版本 - 需要重新安装CUDA版本")

    print(f"PyTorch编译选项: {torch.__config__.show()}")


if __name__ == "__main__":
    has_gpu = check_gpu_detailed()
    check_pytorch_installation()

    if has_gpu:
        print("\n🎉 GPU可用！问题可能在YOLO训练脚本中")
    else:
        print("\n💡 需要安装GPU版本的PyTorch")