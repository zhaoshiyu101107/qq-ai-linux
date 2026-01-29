#venv/bin/env python3
"""
GPU诊断脚本 - 找出为什么脚本检测不到GPU
"""

import torch
import subprocess
import os
import sys

def check_pytorch_gpu():
    """检查PyTorch是否能检测到GPU"""
    print("=" * 60)
    print("🎮 PyTorch GPU检测")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA编译版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'None'}")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    
    if cuda_available:
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"     显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"     算力: {props.major}.{props.minor}")
    else:
        print("❌ PyTorch无法访问GPU")
    
    return cuda_available

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("\n" + "=" * 60)
    print("🛠️  NVIDIA驱动检测")
    print("=" * 60)
    
    try:
        # 尝试运行nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ nvidia-smi命令可用")
            
            # 提取驱动版本
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"  驱动版本: {line.strip()}")
                    break
            
            # 提取GPU信息
            for line in lines:
                if 'NVIDIA' in line and 'GB' in line:
                    print(f"  GPU信息: {line.strip()}")
            
            return True
        else:
            print("❌ nvidia-smi命令失败")
            print(f"  错误: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ nvidia-smi未找到")
        print("  可能原因:")
        print("  1. NVIDIA驱动未安装")
        print("  2. nvidia-smi不在PATH中")
        return False
    except Exception as e:
        print(f"❌ 检查驱动时出错: {e}")
        return False

def check_cuda_installation():
    """检查CUDA安装"""
    print("\n" + "=" * 60)
    print("📦 CUDA安装检测")
    print("=" * 60)
    
    # 检查常见的CUDA路径
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda/bin",
        "/opt/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
    ]
    
    found_cuda = False
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ 找到CUDA目录: {path}")
            found_cuda = True
            
            # 检查nvcc
            nvcc_path = os.path.join(path, "bin", "nvcc")
            if os.path.exists(nvcc_path):
                print(f"  nvcc存在: {nvcc_path}")
            else:
                print(f"  nvcc不存在于: {nvcc_path}")
    
    if not found_cuda:
        print("❌ 未找到CUDA安装目录")
    
    # 检查环境变量
    print("\n📝 环境变量检查:")
    env_vars = ['CUDA_HOME', 'CUDA_PATH', 'PATH']
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        if var == 'PATH':
            print(f"  {var}: (长度: {len(value)} 字符)")
            # 检查PATH中是否包含CUDA
            if 'cuda' in value.lower():
                print(f"    PATH中包含CUDA")
        else:
            print(f"  {var}: {value}")
    
    return found_cuda

def check_pytorch_installation_type():
    """检查PyTorch安装类型（CPU/GPU）"""
    print("\n" + "=" * 60)
    print("🔍 PyTorch安装类型检测")
    print("=" * 60)
    
    # 检查PyTorch是否支持CUDA
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 检查编译选项
        print(f"编译时CUDA支持: {'是' if torch.cuda.is_available() else '否'}")
        
        # 尝试导入cuda模块
        try:
            import torch.cuda
            print(f"torch.cuda模块: 可导入")
            
            # 检查_cuda模块
            if hasattr(torch, '_C'):
                print(f"torch._C存在: 是")
            else:
                print(f"torch._C存在: 否")
                
        except ImportError as e:
            print(f"torch.cuda导入失败: {e}")
            print("⚠️  这可能是CPU版本的PyTorch")
            
    except Exception as e:
        print(f"检查PyTorch时出错: {e}")
    
    # 检查pip安装的包
    print("\n🔧 检查已安装的PyTorch包:")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '|', 'grep', 'torch'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(result.stdout)
    except:
        pass

def check_gpu_with_lspci():
    """使用lspci检查GPU（Linux）"""
    print("\n" + "=" * 60)
    print("💻 系统硬件检测")
    print("=" * 60)
    
    try:
        # 检查lspci（Linux）
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_lines = [line for line in result.stdout.split('\n') 
                        if 'VGA' in line or '3D' in line or 'Display' in line]
            
            if gpu_lines:
                print("✅ 系统检测到显卡:")
                for line in gpu_lines:
                    print(f"  {line}")
            else:
                print("❌ 系统中未检测到显卡设备")
        else:
            print("⚠️  lspci命令不可用")
            
    except FileNotFoundError:
        print("⚠️  lspci命令未找到（可能不是Linux系统）")
    except Exception as e:
        print(f"检查硬件时出错: {e}")

def check_arch_specific():
    """Arch Linux特定检查"""
    print("\n" + "=" * 60)
    print("🐧 Arch Linux特定检查")
    print("=" * 60)
    
    # 检查是否安装了cuda包
    try:
        result = subprocess.run(['pacman', '-Q', 'cuda'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 系统已安装CUDA包")
            print(f"  版本: {result.stdout.strip()}")
        else:
            print("❌ 系统未安装CUDA包")
            
        # 检查nvidia驱动包
        nvidia_packages = ['nvidia', 'nvidia-utils', 'nvidia-settings']
        for pkg in nvidia_packages:
            result = subprocess.run(['pacman', '-Q', pkg], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ 已安装: {pkg}")
            else:
                print(f"❌ 未安装: {pkg}")
                
    except Exception as e:
        print(f"检查Arch包时出错: {e}")

def main():
    """主诊断函数"""
    print("🤖 GPU诊断工具")
    print("=" * 60)
    print("此工具将帮助诊断为什么脚本检测不到GPU")
    print("=" * 60)
    
    # 检查操作系统
    import platform
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    # 运行所有检查
    pytorch_gpu = check_pytorch_gpu()
    nvidia_driver = check_nvidia_driver()
    cuda_installed = check_cuda_installation()
    check_pytorch_installation_type()
    
    if platform.system() == 'Linux':
        check_gpu_with_lspci()
        if 'arch' in platform.platform().lower():
            check_arch_specific()
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 诊断总结")
    print("=" * 60)
    
    issues = []
    
    if not pytorch_gpu:
        issues.append("❌ PyTorch无法检测到GPU")
    if not nvidia_driver:
        issues.append("❌ NVIDIA驱动可能有问题")
    if not cuda_installed:
        issues.append("⚠️  CUDA可能未正确安装")
    
    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 解决方案:")
        print("1. 确保安装了NVIDIA驱动")
        print("2. 安装CUDA工具包")
        print("3. 重新安装GPU版本的PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("✅ 所有检查通过！")
        print("如果脚本仍然检测不到GPU，请检查脚本中的检测逻辑")
    
    print("\n🔧 快速修复命令:")
    print("安装GPU版PyTorch:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()