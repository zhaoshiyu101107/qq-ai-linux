#!/usr/bin/env python3
"""
虚拟环境GPU修复脚本
在已激活的虚拟环境中运行
"""

import subprocess
import sys
import os

def run_cmd(cmd, desc=""):
    """运行命令"""
    if desc:
        print(f"📦 {desc}...")
    
    print(f"   $ {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ 成功")
            if result.stdout.strip():
                # 显示重要信息
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['success', 'installed', 'cuda', 'gpu', 'version']):
                        print(f"      {line}")
            return True, result.stdout
        else:
            print(f"   ❌ 失败")
            if result.stderr:
                # 显示关键错误信息
                lines = result.stderr.split('\n')
                for line in lines[:3]:  # 只显示前3行错误
                    if line.strip():
                        print(f"      {line}")
            return False, result.stderr
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False, str(e)

def check_venv():
    """检查是否在虚拟环境中"""
    print("🔍 检查虚拟环境状态...")
    
    in_venv = sys.prefix != sys.base_prefix
    print(f"   在虚拟环境中: {'✅ 是' if in_venv else '❌ 否'}")
    
    if not in_venv:
        print("\n❌ 请先激活虚拟环境！")
        print("   激活命令示例:")
        print("   source ~/你的虚拟环境路径/bin/activate")
        return False
    
    # 检查Python和pip位置
    success, output = run_cmd("which python", "Python位置")
    success, output = run_cmd("which pip", "pip位置")
    
    return True

def clean_current_installation():
    """清理当前安装"""
    print("\n🧹 清理当前安装...")
    
    # 强制卸载所有相关包
    packages = ['torch', 'torchvision', 'torchaudio', 'torchtext', 'torchdata']
    
    for pkg in packages:
        # 检查是否安装
        success, _ = run_cmd(f"pip show {pkg}", f"检查{pkg}")
        if success:
            # 卸载
            run_cmd(f"pip uninstall -y {pkg}", f"卸载{pkg}")
    
    # 清理pip缓存
    run_cmd("pip cache purge", "清理pip缓存")
    
    print("✅ 清理完成")

def choose_cuda_version():
    """选择CUDA版本"""
    print("\n🎯 选择CUDA版本")
    print("="*50)
    
    # 检查nvidia-smi支持的CUDA版本
    success, output = run_cmd("nvidia-smi | grep 'CUDA Version'", "检查驱动支持的CUDA")
    
    if success:
        # 解析CUDA版本
        import re
        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', output)
        if match:
            driver_cuda = match.group(1)
            print(f"   驱动支持: CUDA {driver_cuda}")
    
    print("\n   PyTorch官方预编译版本:")
    print("   [1] CUDA 11.8 - 最稳定，兼容性最好 (推荐)")
    print("   [2] CUDA 12.1 - 较新，性能较好")
    print("   [3] CUDA 12.4 - 最新")
    
    while True:
        choice = input("\n   选择版本 (1-3, 默认: 1): ").strip()
        
        if not choice:
            choice = "1"
        
        versions = {
            "1": ("11.8", "https://download.pytorch.org/whl/cu118"),
            "2": ("12.1", "https://download.pytorch.org/whl/cu121"),
            "3": ("12.4", "https://download.pytorch.org/whl/cu124"),
        }
        
        if choice in versions:
            cuda_ver, index_url = versions[choice]
            print(f"   ✅ 选择: CUDA {cuda_ver}")
            return cuda_ver, index_url
        else:
            print("   ❌ 无效选择，请输入 1-3")

def install_gpu_pytorch():
    """安装GPU版本PyTorch"""
    print("\n🚀 安装GPU版本PyTorch")
    print("="*50)
    
    # 选择CUDA版本
    cuda_ver, index_url = choose_cuda_version()
    
    # 安装命令
    print(f"\n1. 安装CUDA {cuda_ver}版本的PyTorch...")
    
    install_cmd = f"pip install torch torchvision torchaudio --index-url {index_url}"
    success, output = run_cmd(install_cmd, "安装PyTorch")
    
    if not success:
        # 尝试使用清华镜像
        print("\n⚠️ 官方源失败，尝试清华镜像...")
        install_cmd = f"pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple"
        success, output = run_cmd(install_cmd, "使用清华镜像安装")
    
    return success

def verify_installation():
    """验证安装"""
    print("\n✅ 验证安装")
    print("="*50)
    
    # 检查Python导入
    check_code = '''
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"编译选项: {torch.__config__.show()}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🎉 GPU加速已启用！")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"      显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"      算力: {props.major}.{props.minor}")
else:
    print("❌ GPU不可用")
    print("可能原因:")
    print("1. 安装的仍然是CPU版本")
    print("2. CUDA版本不匹配")
    print("3. 驱动问题")
'''
    
    # 写入临时文件并执行
    with open("/tmp/check_gpu.py", "w") as f:
        f.write(check_code)
    
    success, output = run_cmd("python /tmp/check_gpu.py", "验证GPU支持")
    
    # 清理临时文件
    run_cmd("rm -f /tmp/check_gpu.py")
    
    return success and "CUDA可用: True" in output

def install_ai_dependencies():
    """安装AI依赖"""
    print("\n🤖 安装AI模型依赖")
    print("="*50)
    
    dependencies = [
        "transformers>=4.36.0",
        "accelerate>=0.24.0",
        "sentencepiece",
        "protobuf",
        "einops",
        "tiktoken",
        "huggingface-hub",
    ]
    
    for dep in dependencies:
        run_cmd(f"pip install {dep}", f"安装{dep.split('>=')[0]}")
    
    print("✅ AI依赖安装完成")

def main():
    """主函数"""
    print("🤖 虚拟环境GPU修复工具")
    print("="*60)
    print("注意: 请确保已经激活了虚拟环境")
    print("="*60)
    
    # 检查虚拟环境
    if not check_venv():
        return
    
    # 1. 清理
    clean_current_installation()
    
    # 2. 安装GPU PyTorch
    if not install_gpu_pytorch():
        print("❌ PyTorch安装失败")
        return
    
    # 3. 验证
    if not verify_installation():
        print("❌ GPU验证失败")
        return
    
    # 4. 安装AI依赖
    install_ai_dependencies()
    
    print("\n" + "="*60)
    print("🎉 修复完成！")
    print("="*60)
    print("虚拟环境中的GPU支持已启用")
    print("\n💡 使用说明:")
    print("1. 下次使用时，先激活虚拟环境:")
    print("   source ~/你的虚拟环境路径/bin/activate")
    print("2. 运行你的AI脚本:")
    print("   python main.py")
    print("="*60)

if __name__ == "__main__":
    main()
