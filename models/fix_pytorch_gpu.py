#!/usr/bin/env python3
"""
PyTorch GPU版本修复脚本
专门针对Arch Linux和CPU版本问题
"""

import subprocess
import sys
import os

def run_command(cmd, desc=""):
    """运行命令并显示进度"""
    if desc:
        print(f"📦 {desc}...")
    
    print(f"   $ {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"   ✅ 成功")
            if result.stdout.strip():
                print(f"     输出: {result.stdout[:200]}...")
            return True
        else:
            print(f"   ❌ 失败")
            if result.stderr:
                print(f"     错误: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏳ 超时，正在继续...")
        return True
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False

def check_current_pytorch():
    """检查当前PyTorch版本"""
    print("🔍 检查当前PyTorch安装...")
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'import torch; print(torch.__version__)'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   当前版本: {version}")
            
            # 检查是否是CPU版本
            if '+cpu' in version.lower():
                print("   ❌ 检测到CPU版本的PyTorch")
                return True, version  # True表示需要修复
            else:
                print("   ✅ 已经是GPU版本")
                return False, version
        else:
            print("   ❓ 无法获取版本")
            return True, "unknown"
            
    except Exception as e:
        print(f"   ❌ 检查失败: {e}")
        return True, "error"

def check_cuda_driver():
    """检查CUDA驱动"""
    print("\n🎮 检查CUDA驱动...")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # 解析nvidia-smi输出
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    driver = line.split('Driver Version:')[-1].split()[0]
                    print(f"   驱动版本: {driver}")
                
                if 'CUDA Version' in line:
                    cuda = line.split('CUDA Version:')[-1].split()[0]
                    print(f"   支持CUDA版本: {cuda}")
            
            print("   ✅ NVIDIA驱动正常")
            return True
        else:
            print("   ❌ nvidia-smi失败")
            return False
            
    except FileNotFoundError:
        print("   ❌ nvidia-smi未找到")
        return False
    except Exception as e:
        print(f"   ❌ 检查失败: {e}")
        return False

def get_cuda_version_from_driver():
    """从驱动获取CUDA版本"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            # 简化：根据驱动版本推测CUDA版本
            driver_major = int(driver_version.split('.')[0])
            
            # 驱动版本到CUDA版本的映射（简化）
            if driver_major >= 580:  # 580.x支持CUDA 12.x
                return "12.1"
            elif driver_major >= 525:  # 525.x支持CUDA 11.8
                return "11.8"
            elif driver_major >= 470:  # 470.x支持CUDA 11.4
                return "11.7"
            else:
                return "11.8"  # 默认
                
        return "11.8"  # 默认
        
    except:
        return "11.8"  # 默认

def install_gpu_pytorch():
    """安装GPU版本的PyTorch"""
    print("\n🚀 安装GPU版本PyTorch")
    print("="*60)
    
    # 1. 卸载当前版本
    print("1️⃣  卸载当前CPU版本...")
    success = run_command(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 
         'torch', 'torchvision', 'torchaudio'],
        "卸载PyTorch"
    )
    
    if not success:
        print("⚠️  卸载可能失败，尝试强制卸载...")
        run_command([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torch'])
    
    # 2. 根据驱动选择CUDA版本
    print("\n2️⃣  选择CUDA版本...")
    cuda_version = get_cuda_version_from_driver()
    print(f"   根据驱动选择: CUDA {cuda_version}")
    
    # 询问用户确认
    versions = {
        "1": ("11.8", "https://download.pytorch.org/whl/cu118"),
        "2": ("12.1", "https://download.pytorch.org/whl/cu121"),
        "3": ("12.4", "https://download.pytorch.org/whl/cu124"),
    }
    
    print("\n   可选版本:")
    for key, (ver, url) in versions.items():
        print(f"   [{key}] CUDA {ver}")
    
    choice = input(f"\n   选择版本 (1-3, 默认 {cuda_version}): ").strip()
    
    if choice in versions:
        selected_ver, index_url = versions[choice]
    else:
        # 使用默认版本
        for ver, url in versions.values():
            if ver == cuda_version:
                selected_ver, index_url = cuda_version, url
                break
        else:
            selected_ver, index_url = versions["1"]  # 默认11.8
    
    print(f"   使用: CUDA {selected_ver} ({index_url})")
    
    # 3. 安装GPU版本
    print(f"\n3️⃣  安装CUDA {selected_ver} 版本的PyTorch...")
    
    # 对于Arch Linux，可能需要--break-system-packages
    is_arch = os.path.exists('/etc/arch-release')
    
    pip_cmd = [sys.executable, '-m', 'pip', 'install']
    
    if is_arch:
        pip_cmd.append('--break-system-packages')
        print("   🐧 检测到Arch Linux，使用--break-system-packages")
    
    pip_cmd.extend(['torch', 'torchvision', 'torchaudio'])
    pip_cmd.extend(['--index-url', index_url])
    
    success = run_command(pip_cmd, "安装GPU版本")
    
    if not success:
        print("\n⚠️  安装失败，尝试使用清华镜像源...")
        
        # 尝试清华镜像
        pip_cmd = [sys.executable, '-m', 'pip', 'install']
        if is_arch:
            pip_cmd.append('--break-system-packages')
        
        pip_cmd.extend(['torch', 'torchvision', 'torchaudio', '-i', 
                       'https://pypi.tuna.tsinghua.edu.cn/simple'])
        
        run_command(pip_cmd, "使用清华镜像安装")
    
    return success

def verify_installation():
    """验证安装"""
    print("\n✅ 验证安装...")
    
    try:
        # 运行Python代码检查
        check_code = """
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"      显存: {props.total_memory / 1024**3:.1f} GB")
else:
    print("❌ GPU仍然不可用")
"""
        
        result = subprocess.run(
            [sys.executable, '-c', check_code],
            capture_output=True, text=True
        )
        
        print(result.stdout)
        
        if torch.cuda.is_available() in result.stdout:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def main():
    print("🤖 PyTorch GPU版本修复工具")
    print("="*60)
    
    # 检查当前安装
    need_fix, current_version = check_current_pytorch()
    
    if not need_fix and '+cpu' not in current_version.lower():
        print("\n✅ 当前已经是GPU版本，无需修复")
        return
    
    # 检查驱动
    driver_ok = check_cuda_driver()
    
    if not driver_ok:
        print("\n❌ NVIDIA驱动有问题，请先安装驱动:")
        print("   sudo pacman -S nvidia nvidia-utils nvidia-settings")
        return
    
    # 询问用户是否继续
    print("\n" + "="*60)
    response = input("是否安装GPU版本的PyTorch？(Y/n): ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        # 安装
        success = install_gpu_pytorch()
        
        if success:
            # 验证
            verify_installation()
            
            print("\n" + "="*60)
            print("🎉 修复完成！")
            print("现在可以运行你的AI脚本了")
            print("="*60)
        else:
            print("\n❌ 安装失败")
            print("请尝试手动安装:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("\n❌ 用户取消")
    
    print("\n💡 运行以下命令测试GPU:")
    print("  python -c \"import torch; print(f'GPU可用: {torch.cuda.is_available()}')\"")

if __name__ == "__main__":
    main()