#!/bin/bash
# AI项目完整安装和启动脚本
# 支持虚拟环境自动创建、GPU检测、PyTorch版本选择、AI模型下载

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 打印分隔线
print_separator() {
    printf '=%.0s' {1..60}
    echo ""
}

echo -e "${GREEN}🤖 AI项目启动${NC}"
print_separator

# 获取项目路径
PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "项目目录: $PROJECT_PATH"

# 虚拟环境路径
VENV_PATH="${PROJECT_PATH}/venv"

# 检查脚本参数
INSTALL_MODE=false
CHECK_ONLY=false
WEB_MODE=false
API_MODE=false
MODEL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --install)
            INSTALL_MODE=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --web)
            WEB_MODE=true
            shift
            ;;
        --api)
            API_MODE=true
            shift
            ;;
        --model)
            MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --install     重新安装环境和依赖"
            echo "  --check       仅检查环境"
            echo "  --web         启动Web界面"
            echo "  --api         启动API服务"
            echo "  --model <name> 指定要使用的模型"
            echo "  --help        显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查并创建虚拟环境
if [ ! -f "$VENV_PATH/bin/activate" ] || [ "$INSTALL_MODE" = true ]; then
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        echo -e "${YELLOW}虚拟环境不存在: $VENV_PATH${NC}"
    else
        echo -e "${YELLOW}重新安装模式，将重新设置虚拟环境${NC}"
    fi
    
    # 询问用户确认
    if [ ! -f "$VENV_PATH/bin/activate" ]; then
        read -p "是否创建虚拟环境？(y/n, 默认: y): " create_venv
        create_venv=${create_venv:-y}
    else
        read -p "将重新创建虚拟环境，现有环境将被覆盖。继续？(y/n, 默认: n): " create_venv
        create_venv=${create_venv:-n}
    fi
    
    if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
        # 检查Python3是否可用
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}❌ 未找到python3，请先安装Python3。${NC}"
            echo "访问 https://www.python.org/downloads/ 获取安装包"
            exit 1
        fi
        
        # 检查Python版本
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        echo "Python版本: $PYTHON_VERSION"
        
        # Python 3.14可能太新，PyTorch可能没有预编译包
        if [[ "$PYTHON_VERSION" =~ ^3\.1[4-9] ]]; then
            echo -e "${YELLOW}⚠️  注意: Python 3.14+ 可能太新，PyTorch可能没有预编译包${NC}"
            echo -e "${YELLOW}建议使用 Python 3.8-3.11 以获得最佳兼容性${NC}"
            read -p "是否继续？(y/n, 默认: y): " continue_install
            continue_install=${continue_install:-y}
            if [[ $continue_install != "y" && $continue_install != "Y" ]]; then
                exit 1
            fi
        fi
        
        # 删除现有虚拟环境
        if [ -d "$VENV_PATH" ]; then
            rm -rf "$VENV_PATH"
        fi
        
        echo "创建虚拟环境..."
        python3 -m venv "$VENV_PATH"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ 虚拟环境创建成功: $VENV_PATH${NC}"
        else
            echo -e "${RED}❌ 虚拟环境创建失败${NC}"
            echo "可能需要安装python3-venv包:"
            echo "Ubuntu/Debian: sudo apt-get install python3-venv"
            echo "CentOS/RHEL: sudo yum install python3-venv"
            echo "macOS: 确保已安装Python3"
            exit 1
        fi
    else
        echo -e "${YELLOW}❌ 取消创建虚拟环境${NC}"
        if [ ! -f "$VENV_PATH/bin/activate" ]; then
            exit 1
        fi
    fi
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source "$VENV_PATH/bin/activate"

# 检查Python版本
echo "Python版本: $(python --version)"

# 在安装模式下升级pip和安装基础依赖
if [ ! -f "$VENV_PATH/.installed" ] || [ "$INSTALL_MODE" = true ]; then
    echo -e "${YELLOW}[1/4] 升级pip...${NC}"
    pip install --upgrade pip
    
    echo -e "${YELLOW}[2/4] 检测系统环境...${NC}"
    
    # 检测GPU和CUDA
    HAS_NVIDIA=false
    HAS_CUDA=false
    CUDA_VERSION=""
    
    # 检查nvidia-smi命令
    if command -v nvidia-smi &> /dev/null; then
        HAS_NVIDIA=true
        echo -e "${GREEN}✓ 检测到NVIDIA GPU${NC}"
        
        # 检查CUDA版本
        if nvidia-smi --query | grep -q "CUDA Version"; then
            HAS_CUDA=true
            CUDA_VERSION=$(nvidia-smi --query | grep "CUDA Version" | awk '{print $NF}' | cut -d'.' -f1)
            echo -e "${GREEN}✓ 检测到CUDA版本: ${CUDA_VERSION}${NC}"
        elif nvidia-smi | grep -q "CUDA Version"; then
            HAS_CUDA=true
            CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | cut -d'.' -f1)
            echo -e "${GREEN}✓ 检测到CUDA版本: ${CUDA_VERSION}${NC}"
        else
            echo -e "${YELLOW}⚠ NVIDIA驱动已安装但未检测到CUDA版本${NC}"
            # 尝试从nvcc获取CUDA版本
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c1-2)
                echo -e "${GREEN}✓ 从nvcc检测到CUDA版本: ${CUDA_VERSION}${NC}"
                HAS_CUDA=true
            fi
        fi
    else
        echo -e "${YELLOW}ℹ 未检测到NVIDIA GPU或驱动${NC}"
    fi
    
    # 询问用户选择PyTorch版本
    echo -e "\n${BLUE}请选择PyTorch安装版本:${NC}"
    
    if [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = true ]; then
        echo "1. GPU加速版 - 最佳性能，需要NVIDIA GPU"
        echo "2. CPU版 - 仅CPU，无GPU加速"
        echo "3. CPU+GPU通用版 - 智能切换，有GPU时用GPU，无GPU时用CPU"
        echo "4. 自动选择 - 根据系统自动选择最佳版本"
        
        read -p "请选择 [1/2/3/4] (默认: 4): " choice
        
        case ${choice:-4} in
            1)
                PYTORCH_VERSION="gpu"
                echo -e "${YELLOW}选择: GPU加速版本${NC}"
                ;;
            2)
                PYTORCH_VERSION="cpu"
                echo -e "${YELLOW}选择: CPU版本${NC}"
                ;;
            3)
                PYTORCH_VERSION="universal"
                echo -e "${YELLOW}选择: CPU+GPU通用版本${NC}"
                ;;
            4)
                PYTORCH_VERSION="auto"
                echo -e "${YELLOW}选择: 自动选择版本${NC}"
                ;;
        esac
    else
        echo "1. CPU版 - 仅CPU，无GPU加速"
        echo "2. CPU+GPU通用版 - 智能切换，有GPU时用GPU，无GPU时用CPU"
        echo "3. 自动选择 - 根据系统自动选择最佳版本"
        read -p "请选择 [1/2/3] (默认: 3): " choice
        
        case ${choice:-3} in
            1)
                PYTORCH_VERSION="cpu"
                echo -e "${YELLOW}选择: CPU版本${NC}"
                ;;
            2)
                PYTORCH_VERSION="universal"
                echo -e "${YELLOW}选择: CPU+GPU通用版本${NC}"
                ;;
            3)
                PYTORCH_VERSION="auto"
                echo -e "${YELLOW}选择: 自动选择版本${NC}"
                ;;
        esac
    fi
    
    # 根据选择安装PyTorch
    echo -e "\n${YELLOW}[3/4] 安装PyTorch...${NC}"
    
    if [ "$PYTORCH_VERSION" = "auto" ]; then
        if [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = true ]; then
            echo "自动选择: 安装GPU版本"
            PYTORCH_VERSION="gpu"
        else
            echo "自动选择: 安装CPU版本"
            PYTORCH_VERSION="cpu"
        fi
    fi
    
    # 显示PyTorch安装信息
    echo "安装PyTorch..."
    echo "这可能需要几分钟时间，请耐心等待..."
    
    # 先安装一些基础依赖
    pip install numpy
    
    case $PYTORCH_VERSION in
        "cpu")
            echo "安装PyTorch CPU版本..."
            echo "尝试从PyTorch官方源安装..."
            if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir; then
                echo -e "${YELLOW}PyTorch官方源安装失败，尝试使用pip默认源...${NC}"
                pip install torch torchvision torchaudio --no-cache-dir
            fi
            ;;
        "gpu")
            echo "安装PyTorch GPU版本..."
            echo "尝试多种安装方式..."
            
            # 方法1: 尝试使用PyTorch官方的最新稳定版
            echo -e "\n尝试方法1: PyTorch官方最新稳定版..."
            if pip install torch torchvision torchaudio --no-cache-dir; then
                echo -e "${GREEN}✅ 方法1成功${NC}"
            else
                echo -e "${YELLOW}方法1失败，尝试方法2...${NC}"
                
                # 方法2: 尝试使用CUDA 11.8版本
                echo "尝试方法2: CUDA 11.8版本..."
                if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir; then
                    echo -e "${GREEN}✅ 方法2成功${NC}"
                else
                    echo -e "${YELLOW}方法2失败，尝试方法3...${NC}"
                    
                    # 方法3: 尝试使用CUDA 12.1版本
                    echo "尝试方法3: CUDA 12.1版本..."
                    if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir; then
                        echo -e "${GREEN}✅ 方法3成功${NC}"
                    else
                        echo -e "${YELLOW}方法3失败，回退到CPU版本...${NC}"
                        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
                    fi
                fi
            fi
            ;;
        "universal")
            echo "安装PyTorch CPU+GPU通用版本..."
            echo "将安装支持GPU的版本，即使没有GPU也能在CPU上运行..."
            
            # 首先尝试安装标准版本（通常包含CPU和GPU支持）
            if pip install torch torchvision torchaudio --no-cache-dir; then
                echo -e "${GREEN}✅ 安装成功 - 通用版本${NC}"
                echo "此版本支持："
                echo "- 有GPU时自动使用GPU加速"
                echo "- 无GPU时自动回退到CPU运行"
            else
                echo -e "${YELLOW}标准版本安装失败，尝试其他方法...${NC}"
                
                # 根据系统是否有CUDA来选择
                if [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = true ]; then
                    echo "系统有CUDA，尝试安装GPU版本..."
                    if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir; then
                        echo -e "${GREEN}✅ GPU版本安装成功${NC}"
                    else
                        echo -e "${YELLOW}GPU版本安装失败，安装CPU版本...${NC}"
                        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
                    fi
                else
                    echo "系统无CUDA，安装CPU版本..."
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
                fi
            fi
            ;;
    esac
    
    # 检查PyTorch安装是否成功
    if python -c "import torch" 2>/dev/null; then
        echo -e "${GREEN}✅ PyTorch安装成功${NC}"
    else
        echo -e "${RED}❌ PyTorch安装失败${NC}"
        echo "可能的原因:"
        echo "1. Python版本太新（如3.14+），PyTorch可能还没有预编译包"
        echo "2. 网络问题导致下载失败"
        echo "3. 系统架构不兼容"
        echo ""
        echo "解决方案:"
        echo "1. 使用较旧的Python版本（如3.8-3.11）"
        echo "2. 检查网络连接"
        echo "3. 尝试手动安装: pip install torch torchvision torchaudio"
        exit 1
    fi
    
    # 安装AI库
    echo -e "\n${YELLOW}[4/4] 安装AI库...${NC}"
    echo "安装Transformers和其他AI库..."
    
    # 逐个安装，以便更好地处理错误
    for package in transformers accelerate sentencepiece protobuf einops tiktoken; do
        echo "安装 $package..."
        pip install "$package" --no-cache-dir || echo -e "${YELLOW}⚠️  $package 安装失败，继续安装其他包...${NC}"
    done
    
    # 标记为已安装
    touch "$VENV_PATH/.installed"
    
    # 保存配置信息
    cat > "$VENV_PATH/config.json" << EOF
{
  "install_date": "$(date '+%Y-%m-%d %H:%M:%S')",
  "pytorch_version": "${PYTORCH_VERSION}",
  "has_nvidia": ${HAS_NVIDIA},
  "has_cuda": ${HAS_CUDA},
  "cuda_version": "${CUDA_VERSION:-null}",
  "python_version": "$(python --version | awk '{print $2}')"
}
EOF
    
    echo -e "${GREEN}✅ 所有依赖安装完成！${NC}"
fi

# 检查PyTorch是否正常工作
echo -n "检查PyTorch安装状态..."
if ! python -c "import torch" 2>/dev/null; then
    echo -e "\n${RED}❌ PyTorch未正确安装${NC}"
    echo "正在尝试重新安装PyTorch..."
    
    echo "尝试安装最新版本..."
    pip install torch torchvision torchaudio --no-cache-dir
    
    if ! python -c "import torch" 2>/dev/null; then
        echo -e "${RED}❌ PyTorch安装失败${NC}"
        echo "请尝试以下解决方案:"
        echo "1. 降低Python版本到3.8-3.11"
        echo "2. 手动安装: pip install torch torchvision torchaudio"
        echo "3. 查看错误信息并搜索解决方案"
        exit 1
    fi
else
    echo -e "${GREEN}✅ PyTorch已安装${NC}"
fi

# 检查GPU状态
echo "检查GPU状态..."
if python -c "import torch" 2>/dev/null; then
    python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

# 检测是否安装了GPU版本的PyTorch
has_cuda_built = torch.backends.cuda.is_built()
has_cuda_support = torch.cuda.is_available()

print(f'PyTorch是否编译了CUDA支持: {has_cuda_built}')

if has_cuda_built and has_cuda_support:
    print(f'🎮 GPU加速已启用 ({torch.cuda.device_count()}个GPU)')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
elif has_cuda_built and not has_cuda_support:
    print('🔧 安装了GPU版本的PyTorch，但CUDA当前不可用')
    print('💡 可能原因: CUDA驱动不匹配或未安装CUDA工具包')
    print('💡 系统将使用CPU运行，但模型支持GPU加速')
else:
    print('💻 使用CPU版本PyTorch运行')
"
else
    echo -e "${RED}❌ PyTorch未正确安装${NC}"
    exit 1
fi

# 如果只需要检查环境
if [ "$CHECK_ONLY" = true ]; then
    print_separator
    echo "环境检查完成！"
    exit 0
fi

# 询问用户选择模型（如果未通过参数指定且未安装过）
MODEL_CONFIG="${PROJECT_PATH}/model_config.json"
if [ -z "$MODEL_OVERRIDE" ] && [ ! -f "$MODEL_CONFIG" ]; then
    echo -e "\n${BLUE}请选择要使用的AI模型:${NC}"
    echo "1. Qwen/Qwen3-0.5B-Instruct (轻量级, 约0.5GB)"
    echo "2. Qwen/Qwen3-0.6B-Instruct (推荐, 约1.2GB)"
    echo "3. Qwen/Qwen3-1.8B-Instruct (平衡, 约3.6GB)"
    echo "4. Qwen/Qwen2.5-0.5B-Instruct (新版, 约0.5GB)"
    echo "5. microsoft/phi-2 (微软Phi-2, 约2.7GB)"
    echo "6. TinyLlama/TinyLlama-1.1B-Chat-v1.0 (小羊驼, 约2.2GB)"
    echo "7. 自定义模型 (输入完整的HuggingFace模型路径)"
    
    read -p "请选择 [1-7] (默认: 2): " model_choice
    
    case ${model_choice:-2} in
        1)
            MODEL_NAME="Qwen/Qwen3-0.5B-Instruct"
            MODEL_SIZE="约0.5GB"
            ;;
        3)
            MODEL_NAME="Qwen/Qwen3-1.8B-Instruct"
            MODEL_SIZE="约3.6GB"
            ;;
        4)
            MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
            MODEL_SIZE="约0.5GB"
            ;;
        5)
            MODEL_NAME="microsoft/phi-2"
            MODEL_SIZE="约2.7GB"
            ;;
        6)
            MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            MODEL_SIZE="约2.2GB"
            ;;
        7)
            read -p "请输入完整的HuggingFace模型路径: " custom_model
            if [ -n "$custom_model" ]; then
                MODEL_NAME="$custom_model"
                MODEL_SIZE="未知大小"
                echo -e "${YELLOW}使用自定义模型: ${MODEL_NAME}${NC}"
            else
                MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
                MODEL_SIZE="约1.2GB"
                echo -e "${YELLOW}使用默认模型: ${MODEL_NAME}${NC}"
            fi
            ;;
        2)
            MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
            MODEL_SIZE="约1.2GB"
            ;;
    esac
    
    # 保存模型配置
    cat > "$MODEL_CONFIG" << EOF
{
  "model": "${MODEL_NAME}",
  "model_size": "${MODEL_SIZE}",
  "selected_date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
elif [ -n "$MODEL_OVERRIDE" ]; then
    # 使用命令行参数指定的模型
    MODEL_NAME="$MODEL_OVERRIDE"
    MODEL_SIZE="未知大小"
    
    cat > "$MODEL_CONFIG" << EOF
{
  "model": "${MODEL_NAME}",
  "model_size": "${MODEL_SIZE}",
  "selected_date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
    echo -e "${YELLOW}使用指定模型: ${MODEL_NAME}${NC}"
else
    # 读取现有配置
    if [ -f "$MODEL_CONFIG" ]; then
        if command -v python &> /dev/null; then
            MODEL_NAME=$(python -c "
import json
try:
    with open('$MODEL_CONFIG', 'r') as f:
        data = json.load(f)
    print(data.get('model', 'Qwen/Qwen3-0.6B-Instruct'))
except:
    print('Qwen/Qwen3-0.6B-Instruct')
" 2>/dev/null)
            MODEL_SIZE=$(python -c "
import json
try:
    with open('$MODEL_CONFIG', 'r') as f:
        data = json.load(f)
    print(data.get('model_size', '约1.2GB'))
except:
    print('约1.2GB')
" 2>/dev/null)
        else
            MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
            MODEL_SIZE="约1.2GB"
        fi
        echo -e "${GREEN}使用已配置模型: ${MODEL_NAME}${NC}"
    else
        MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
        MODEL_SIZE="约1.2GB"
    fi
fi

# 进入项目目录
cd "$PROJECT_PATH"

print_separator
echo "项目目录: $PROJECT_PATH"
echo "虚拟环境: $VENV_PATH"
echo "选择模型: $MODEL_NAME ($MODEL_SIZE)"
print_separator

# 创建主运行脚本（如果不存在）
if [ ! -f "run_ai.py" ]; then
    echo "创建主运行脚本..."
    cat > run_ai.py << 'EOF'
#!/usr/bin/env python3
"""
AI模型本地运行脚本
支持多种模型，完全本地运行
支持CPU/GPU自动切换
"""

import os
import sys
import torch
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def check_gpu_info():
    """检查GPU信息"""
    config = load_config()
    
    print("=" * 50)
    print("🤖 本地AI环境")
    print("=" * 50)
    print(f"目录: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"模型: {config.get('model', 'Qwen/Qwen3-0.6B-Instruct')}")
    print(f"模型大小: {config.get('model_size', '约1.2GB')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    cuda_built = torch.backends.cuda.is_built()
    
    print(f"CUDA可用: {cuda_available}")
    print(f"PyTorch是否编译了CUDA支持: {cuda_built}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(i)
                print(f"    内存: {props.total_memory / 1024**3:.2f} GB")
                print(f"    算力: {props.major}.{props.minor}")
    else:
        if cuda_built:
            print("🔧 安装了GPU版本的PyTorch，但CUDA当前不可用")
            print("💡 系统将使用CPU运行")
        else:
            print("💻 使用CPU版本PyTorch运行")
    
    print("=" * 50)
    print()

def load_model():
    """加载AI模型，自动选择最佳设备"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    config = load_config()
    model_name = config.get('model', 'Qwen/Qwen3-0.6B-Instruct')
    model_size = config.get('model_size', '约1.2GB')
    
    print(f"正在加载模型: {model_name}")
    print(f"模型大小: {model_size}")
    print("首次运行需要下载模型文件")
    print("下载完成后会缓存，下次无需重新下载")
    print()
    
    # 设置模型缓存目录到本地
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 加载tokenizer
        print("1. 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # 加载模型
        print("2. 加载模型...")
        
        # 检查是否有GPU可用
        if torch.cuda.is_available():
            print("  检测到GPU，使用GPU加速")
            print(f"  可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # 根据模型大小和可用内存选择加载方式
            model_size_mb = 0
            if "0.5B" in model_name:
                model_size_mb = 500
            elif "0.6B" in model_name:
                model_size_mb = 1200
            elif "1.8B" in model_name:
                model_size_mb = 3600
            elif "4B" in model_name:
                model_size_mb = 8000
            
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if model_size_mb > 0 and gpu_memory_gb * 1024 < model_size_mb * 1.5:
                print(f"  ⚠️  GPU内存可能不足 ({gpu_memory_gb:.1f} GB < 建议 {model_size_mb/1024*1.5:.1f} GB)")
                print(f"  使用混合精度加载以节省内存...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
        else:
            print("  未检测到GPU，使用CPU运行")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        
        print(f"  模型加载到设备: {model.device}")
        return tokenizer, model
        
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        print("\n💡 尝试解决方案:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 检查模型名称是否正确")
        print("4. 如果内存不足，尝试更小的模型")
        return None, None

def chat_loop(tokenizer, model):
    """对话循环"""
    print()
    print("✅ 模型加载成功！")
    print("-" * 50)
    print(f"运行设备: {model.device}")
    print("💡 提示: 输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！👋")
                break
                
            if not user_input:
                continue
            
            # 准备对话格式
            messages = [
                {"role": "system", "content": "你是一个乐于助人的AI助手。"},
                {"role": "user", "content": user_input}
            ]
            
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # 如果模型不支持chat_template，使用简单格式
                text = f"用户: {user_input}\n助手:"
            
            # 编码并生成
            model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            print("思考中...", end="", flush=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            print("\r" + " " * 20, end="\r")  # 清除"思考中..."
            
            # 解码回复
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取模型回复部分
            if "助手:" in response:
                response = response.split("助手:")[-1].strip()
            elif "assistant" in response:
                response = response.split("assistant")[-1].strip()
            elif "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            print(f"AI: {response}")
            
        except KeyboardInterrupt:
            print("\n\n退出程序")
            break
        except torch.cuda.OutOfMemoryError:
            print("\n❌ GPU内存不足！")
            print("💡 建议:")
            print("1. 关闭其他使用GPU的程序")
            print("2. 使用更小的模型")
            print("3. 重启程序")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue

def main():
    import argparse
    parser = argparse.ArgumentParser(description="运行本地AI模型")
    parser.add_argument("--web", action="store_true", help="启动Web界面")
    parser.add_argument("--api", action="store_true", help="启动API服务")
    parser.add_argument("--cpu-only", action="store_true", help="强制使用CPU")
    
    args = parser.parse_args()
    
    check_gpu_info()
    
    # 如果指定了--cpu-only，强制使用CPU
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("🔧 强制使用CPU模式")
    
    # 加载模型
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        return
    
    if args.web:
        # 启动Web界面
        try:
            import gradio as gr
            print("启动Web界面...")
            
            def respond(message, history):
                inputs = tokenizer(message, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
            
            gr.ChatInterface(respond).launch(server_name="0.0.0.0", server_port=7860)
        except ImportError:
            print("未安装gradio，无法启动Web界面")
            print("请运行: pip install gradio")
            chat_loop(tokenizer, model)
    elif args.api:
        # 启动API服务
        try:
            from fastapi import FastAPI
            import uvicorn
            print("启动API服务...")
            
            app = FastAPI()
            
            @app.post("/chat")
            async def chat_endpoint(message: dict):
                user_input = message.get("message", "")
                inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return {"response": response}
            
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            print("未安装fastapi和uvicorn，无法启动API服务")
            print("请运行: pip install fastapi uvicorn")
            chat_loop(tokenizer, model)
    else:
        # 启动交互式聊天
        chat_loop(tokenizer, model)

if __name__ == "__main__":
    main()
EOF
fi

print_separator
echo -e "${GREEN}🚀 启动AI应用...${NC}"
echo "输入 'quit' 退出应用"
print_separator

# 运行主程序
ARGS=""
if [ "$WEB_MODE" = true ]; then
    ARGS="$ARGS --web"
elif [ "$API_MODE" = true ]; then
    ARGS="$ARGS --api"
fi

python run_ai.py $ARGS