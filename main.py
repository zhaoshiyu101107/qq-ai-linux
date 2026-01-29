#!/usr/bin/env python3
"""
AI聊天系统 - 带自动依赖安装的主程序
运行此脚本会自动检查并安装所有必需的Python包
"""

import sys
import os
import subprocess
import importlib
import platform
from typing import List, Dict, Tuple

# 预定义颜色
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.END) -> None:
    """打印带颜色的文本"""
    print(f"{color}{text}{Colors.END}")

def print_header(title: str) -> None:
    """打印标题"""
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"🤖 {title}", Colors.CYAN + Colors.BOLD)
    print_colored(f"{'='*60}", Colors.CYAN)

def print_success(text: str) -> None:
    """打印成功信息"""
    print_colored(f"✅ {text}", Colors.GREEN)

def print_warning(text: str) -> None:
    """打印警告信息"""
    print_colored(f"⚠️  {text}", Colors.YELLOW)

def print_error(text: str) -> None:
    """打印错误信息"""
    print_colored(f"❌ {text}", Colors.RED)

def print_info(text: str) -> None:
    """打印信息"""
    print_colored(f"💡 {text}", Colors.BLUE)

def check_python_version() -> bool:
    """检查Python版本"""
    print_header("Python版本检查")
    
    python_version = sys.version_info
    print_info(f"当前Python版本: {sys.version.split()[0]}")
    
    # Python 3.14+ 可能太新，PyTorch可能没有预编译包
    if python_version.major == 3 and python_version.minor >= 14:
        print_warning(f"Python 3.{python_version.minor} 可能太新，PyTorch可能没有预编译包")
        print_warning("建议使用 Python 3.8-3.11 以获得最佳兼容性")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print_error(f"需要Python 3.8+，当前版本: {python_version.major}.{python_version.minor}")
        return False
    
    print_success(f"Python版本符合要求 (3.8+)")
    return True

def get_system_info() -> Dict:
    """获取系统信息"""
    system = platform.system()
    info = {
        'os': system,
        'os_release': platform.release(),
        'arch': platform.machine(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pip_version': None
    }
    
    # 获取pip版本
    try:
        import pip
        info['pip_version'] = pip.__version__
    except:
        pass
    
    return info

def check_pip_installed() -> bool:
    """检查pip是否已安装"""
    try:
        import pip
        print_success(f"pip已安装 (版本: {pip.__version__})")
        return True
    except ImportError:
        print_error("pip未安装")
        return False

def install_pip() -> bool:
    """安装pip"""
    print_header("安装pip")
    
    system_info = get_system_info()
    os_type = system_info['os']
    
    print_info(f"操作系统: {os_type}")
    
    try:
        if os_type == "Linux":
            # Linux系统
            if os.path.exists("/etc/arch-release"):
                print_info("检测到Arch Linux，使用pacman安装pip")
                result = subprocess.run(['sudo', 'pacman', '-Sy', '--noconfirm', 'python-pip'], 
                                      capture_output=True, text=True)
            elif os.path.exists("/etc/debian_version"):
                print_info("检测到Debian/Ubuntu，使用apt安装pip")
                result = subprocess.run(['sudo', 'apt', 'update'], capture_output=True, text=True)
                result = subprocess.run(['sudo', 'apt', 'install', '-y', 'python3-pip'], 
                                      capture_output=True, text=True)
            elif os.path.exists("/etc/redhat-release"):
                print_info("检测到RHEL/Fedora，使用yum/dnf安装pip")
                if subprocess.run(['which', 'dnf'], capture_output=True).returncode == 0:
                    result = subprocess.run(['sudo', 'dnf', 'install', '-y', 'python3-pip'], 
                                          capture_output=True, text=True)
                else:
                    result = subprocess.run(['sudo', 'yum', 'install', '-y', 'python3-pip'], 
                                          capture_output=True, text=True)
            else:
                print_warning("未知Linux发行版，尝试通用方法")
                result = subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'], 
                                      capture_output=True, text=True)
        elif os_type == "Darwin":  # macOS
            print_info("检测到macOS，使用ensurepip安装")
            result = subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'], 
                                  capture_output=True, text=True)
        elif os_type == "Windows":
            print_info("检测到Windows，请手动安装pip")
            print("访问: https://pip.pypa.io/en/stable/installation/")
            return False
        else:
            print_warning(f"未知操作系统: {os_type}")
            result = subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'], 
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("pip安装成功")
            return True
        else:
            print_error(f"pip安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"pip安装过程中出错: {e}")
        return False

def get_required_packages() -> List[Dict]:
    """获取必需的包列表"""
    return [
        {
            'name': 'torch',
            'import_name': 'torch',
            'min_version': '2.0.0',
            'description': 'PyTorch深度学习框架',
            'install_cmd': ['torch', 'torchvision', 'torchaudio'],
            'extra_args': ['--index-url', 'https://download.pytorch.org/whl/cpu']
        },
        {
            'name': 'transformers',
            'import_name': 'transformers',
            'min_version': '4.35.0',
            'description': 'Hugging Face Transformers库',
            'install_cmd': ['transformers']
        },
        {
            'name': 'accelerate',
            'import_name': 'accelerate',
            'min_version': '0.24.0',
            'description': '分布式训练加速库',
            'install_cmd': ['accelerate']
        },
        {
            'name': 'sentencepiece',
            'import_name': 'sentencepiece',
            'min_version': '0.1.99',
            'description': '文本分词器',
            'install_cmd': ['sentencepiece']
        },
        {
            'name': 'protobuf',
            'import_name': 'google.protobuf',
            'min_version': '3.20.0',
            'description': 'Protocol Buffers数据格式',
            'install_cmd': ['protobuf']
        },
        {
            'name': 'einops',
            'import_name': 'einops',
            'min_version': '0.7.0',
            'description': '张量操作库',
            'install_cmd': ['einops']
        },
        {
            'name': 'tiktoken',
            'import_name': 'tiktoken',
            'min_version': '0.5.0',
            'description': 'OpenAI的BPE分词器',
            'install_cmd': ['tiktoken']
        },
        {
            'name': 'huggingface-hub',
            'import_name': 'huggingface_hub',
            'min_version': '0.20.0',
            'description': 'Hugging Face模型仓库',
            'install_cmd': ['huggingface-hub']
        }
    ]

def check_package_installed(package_info: Dict) -> Tuple[bool, str]:
    """检查包是否已安装"""
    try:
        module = importlib.import_module(package_info['import_name'])
        
        # 尝试获取版本
        version = None
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        
        if version:
            # 检查版本是否满足要求
            from packaging import version as pkg_version
            current = pkg_version.parse(version)
            required = pkg_version.parse(package_info['min_version'])
            
            if current >= required:
                return True, f"已安装 (版本: {version})"
            else:
                return False, f"版本过低 ({version} < {package_info['min_version']})"
        else:
            return True, "已安装 (版本未知)"
            
    except ImportError:
        return False, "未安装"
    except Exception as e:
        return False, f"检查失败: {str(e)}"

def get_pip_install_args() -> List[str]:
    """获取pip安装参数"""
    system_info = get_system_info()
    os_type = system_info['os']
    
    # 检查是否在虚拟环境中
    in_venv = sys.prefix != sys.base_prefix
    
    if os_type == "Linux" and os.path.exists("/etc/arch-release") and not in_venv:
        # Arch Linux系统，不在虚拟环境中，需要--break-system-packages
        print_warning("检测到Arch Linux，使用--break-system-packages标志")
        return ["--break-system-packages"]
    elif not in_venv:
        # 不在虚拟环境中，使用--user安装到用户目录
        return ["--user"]
    else:
        # 在虚拟环境中，直接安装
        return []

def install_package(package_info: Dict) -> bool:
    """安装单个包"""
    package_name = package_info['name']
    print_info(f"安装 {package_info['description']} ({package_name})...")
    
    pip_args = get_pip_install_args()
    
    # 构建完整的pip命令
    cmd = [sys.executable, '-m', 'pip', 'install']
    cmd.extend(pip_args)
    
    # 如果是torch，添加额外的包和参数
    if package_name == 'torch':
        cmd.extend(package_info['install_cmd'])
        cmd.extend(package_info.get('extra_args', []))
    else:
        cmd.extend(package_info['install_cmd'])
    
    try:
        # 显示进度
        print(f"  运行命令: {' '.join(cmd)}")
        
        # 执行安装
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success(f"  {package_name} 安装成功")
            return True
        else:
            print_error(f"  {package_name} 安装失败: {result.stderr[:200]}")
            return False
            
    except Exception as e:
        print_error(f"  安装过程中出错: {e}")
        return False

def check_and_install_dependencies() -> bool:
    """检查并安装所有依赖"""
    print_header("依赖检查")
    
    # 1. 检查Python版本
    if not check_python_version():
        return False
    
    # 2. 检查pip
    if not check_pip_installed():
        print_info("尝试安装pip...")
        if not install_pip():
            print_error("无法安装pip，请手动安装")
            return False
    
    # 3. 检查必需的包
    required_packages = get_required_packages()
    missing_packages = []
    
    print_info("检查必需的Python包...")
    
    for package in required_packages:
        installed, status = check_package_installed(package)
        
        if installed:
            print_success(f"  {package['name']}: {status}")
        else:
            print_warning(f"  {package['name']}: {status}")
            missing_packages.append(package)
    
    if not missing_packages:
        print_success("所有依赖已满足！")
        return True
    
    # 4. 询问用户是否安装缺失的包
    print_header("安装缺失依赖")
    print(f"需要安装 {len(missing_packages)} 个包:")
    
    for package in missing_packages:
        print(f"  • {package['name']} ({package['description']})")
    
    while True:
        response = input("\n是否安装这些包？ (Y/n): ").strip().lower()
        
        if response in ['', 'y', 'yes']:
            break
        elif response in ['n', 'no']:
            print_warning("用户选择不安装依赖，程序可能无法正常运行")
            return False
        else:
            print("请输入 Y(是) 或 N(否)")
    
    # 5. 安装缺失的包
    print_info("开始安装...")
    success_count = 0
    
    for package in missing_packages:
        if install_package(package):
            success_count += 1
        else:
            print_warning(f"{package['name']} 安装失败")
    
    # 6. 验证安装
    print_header("验证安装")
    all_installed = True
    
    for package in missing_packages:
        installed, status = check_package_installed(package)
        
        if installed:
            print_success(f"  {package['name']}: 验证通过")
        else:
            print_error(f"  {package['name']}: 安装后仍然缺失")
            all_installed = False
    
    if all_installed:
        print_success(f"成功安装 {success_count}/{len(missing_packages)} 个包")
        return True
    else:
        print_error("部分包安装失败，可能需要手动安装")
        print_info("请运行: pip install " + " ".join([p['name'] for p in missing_packages]))
        return False

def setup_ai_system():
    """设置AI系统（原main函数的内容）"""
    print_header("AI系统初始化")
    
    try:
        # 尝试导入核心模块
        from config.gpu_config import detect_gpus, save_gpu_config
        from config.model_config import list_available_models, print_model_info
        from core.device_manager import DeviceManager
        from core.chat_engine import ChatEngine
        from utils.gpu_utils import check_cuda_version, get_system_info, optimize_for_gpu
        
        print_success("核心模块导入成功")
        
    except ImportError as e:
        print_error(f"导入模块失败: {e}")
        print_info("请确保已安装所有依赖并正确设置项目结构")
        return False
    
    # 获取系统信息
    sys_info = get_system_info()
    print_info(f"操作系统: {sys_info.get('os', '未知')}")
    print_info(f"Python版本: {sys_info.get('python_version', '未知')}")
    
    # 检查CUDA
    cuda_info = check_cuda_version()
    if cuda_info['cuda_available']:
        print_success(f"CUDA可用 (版本: {cuda_info.get('cuda_version', '未知')})")
        optimize_for_gpu()
    
    # GPU配置
    print_header("GPU配置")
    device_manager = DeviceManager()
    has_gpu = device_manager.print_device_info()
    
    if has_gpu:
        gpu_config = device_manager.get_user_choice()
        gpus = detect_gpus()
        config_file = save_gpu_config(gpu_config, gpus)
        print_success(f"GPU配置已保存到: {config_file}")
    else:
        print_warning("未检测到GPU，将使用CPU运行")
    
    # 模型选择
    print_header("选择AI模型")
    models = list_available_models()
    
    for i, model_key in enumerate(models, 1):
        print(f"{i}. {model_key}")
    
    while True:
        try:
            choice = input(f"\n选择模型 (1-{len(models)}, 默认: 1): ").strip()
            
            if not choice:
                choice = "1"
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                model_config = print_model_info(selected)
                
                # 创建对话引擎
                print_header("启动对话系统")
                engine = ChatEngine(selected)
                
                # 开始对话
                engine.interactive_chat()
                
                # 保存历史
                engine.save_history()
                
                print_success("会话完成")
                return True
                
            else:
                print_error(f"请输入 1-{len(models)} 之间的数字")
                
        except ValueError:
            print_error("请输入有效的数字")
        except KeyboardInterrupt:
            print_warning("\n用户中断")
            return False
        except Exception as e:
            print_error(f"启动失败: {e}")
            return False
    
    return True

def main():
    """主函数"""
    print_header("AI聊天系统 - 带自动依赖安装")
    
    try:
        # 1. 检查并安装依赖
        if not check_and_install_dependencies():
            print_warning("依赖检查/安装失败，程序可能无法正常运行")
            
            # 询问是否继续
            response = input("\n是否继续运行？ (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("退出程序")
                return
        
        # 2. 设置和运行AI系统
        print_header("启动AI聊天系统")
        setup_ai_system()
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print_colored("👋 程序被用户中断", Colors.YELLOW)
        print("="*60)
    except Exception as e:
        print_error(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*60)
        print_colored("💡 故障排除建议:", Colors.BLUE)
        print("1. 确保网络连接正常")
        print("2. 尝试手动安装依赖: pip install -r requirements.txt")
        print("3. 检查Python版本 (需要3.8+)")
        print("4. 查看详细错误信息")
        print("="*60)
        
        sys.exit(1)

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements = """# AI聊天系统依赖列表
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
sentencepiece>=0.1.99
protobuf>=3.20.0
einops>=0.7.0
tiktoken>=0.5.0
huggingface-hub>=0.20.0

# 可选依赖（用于更高级的功能）
# gradio>=3.0.0  # Web界面
# streamlit>=1.0.0  # Web界面
# fastapi>=0.100.0  # API服务器
# uvicorn>=0.23.0  # ASGI服务器
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print_success("requirements.txt 文件已创建")

if __name__ == "__main__":
    # 检查是否需要创建requirements.txt
    if not os.path.exists("requirements.txt"):
        print_info("未找到requirements.txt文件")
        response = input("是否创建requirements.txt文件？ (Y/n): ").strip().lower()
        if response in ['', 'y', 'yes']:
            create_requirements_file()
    
    # 运行主程序
    main()
