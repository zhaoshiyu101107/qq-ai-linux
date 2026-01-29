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

# 进度条函数
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percent=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))
    
    printf "\r["
    printf "%${completed}s" "" | tr ' ' '='
    printf "%${remaining}s" "" | tr ' ' ' '
    printf "] %3d%%" $percent
    
    if [ $current -eq $total ]; then
        echo -e " ${GREEN}完成!${NC}"
    fi
}

# 带颜色的进度条
show_colored_progress() {
    local current=$1
    local total=$2
    local message="$3"
    local width=50
    local percent=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))
    
    printf "\r${CYAN}%s:${NC} [" "$message"
    
    # 根据百分比改变颜色
    if [ $percent -lt 30 ]; then
        printf "${RED}"
    elif [ $percent -lt 70 ]; then
        printf "${YELLOW}"
    else
        printf "${GREEN}"
    fi
    
    printf "%${completed}s" "" | tr ' ' '█'
    printf "${NC}%${remaining}s" "" | tr ' ' ' '
    printf "] ${BLUE}%3d%%${NC}" $percent
    
    if [ $current -eq $total ]; then
        echo -e " ${GREEN}✓${NC}"
    fi
}

# 步骤进度条
step_progress() {
    local step_num=$1
    local total_steps=$2
    local step_name="$3"
    
    echo -e "\n${MAGENTA}步骤 ${step_num}/${total_steps}:${NC} ${BLUE}${step_name}${NC}"
    for i in $(seq 1 10); do
        sleep 0.05
        show_colored_progress $i 10 "$step_name"
    done
}

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

# 总步骤数
TOTAL_STEPS=4
CURRENT_STEP=0

# 检查并创建虚拟环境
if [ ! -f "$VENV_PATH/bin/activate" ] || [ "$INSTALL_MODE" = true ]; then
    ((CURRENT_STEP++))
    step_progress $CURRENT_STEP $TOTAL_STEPS "创建虚拟环境"
    
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
        
        echo "正在创建虚拟环境..."
        for i in $(seq 1 20); do
            sleep 0.05
            show_colored_progress $i 20 "创建虚拟环境"
        done
        
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
    ((CURRENT_STEP++))
    step_progress $CURRENT_STEP $TOTAL_STEPS "检测系统环境"
    
    echo -e "${YELLOW}检测系统环境...${NC}"
    
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
    ((CURRENT_STEP++))
    step_progress $CURRENT_STEP $TOTAL_STEPS "安装PyTorch"
    
    echo -e "\n${YELLOW}安装PyTorch...${NC}"
    
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
    echo "正在安装PyTorch..."
    echo "这可能需要几分钟时间，请耐心等待..."
    
    # 先安装一些基础依赖
    echo "安装numpy..."
    for i in $(seq 1 10); do
        sleep 0.05
        show_colored_progress $i 10 "安装numpy"
    done
    pip install numpy --no-cache-dir > /dev/null 2>&1
    
    case $PYTORCH_VERSION in
        "cpu")
            echo "安装PyTorch CPU版本..."
            echo "尝试从PyTorch官方源安装..."
            for i in $(seq 1 30); do
                sleep 0.1
                show_colored_progress $i 30 "安装PyTorch CPU版本"
            done
            if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir > /dev/null 2>&1; then
                echo -e "${YELLOW}PyTorch官方源安装失败，尝试使用pip默认源...${NC}"
                for i in $(seq 1 30); do
                    sleep 0.1
                    show_colored_progress $i 30 "安装PyTorch (备用源)"
                done
                pip install torch torchvision torchaudio --no-cache-dir > /dev/null 2>&1
            fi
            ;;
        "gpu")
            echo "安装PyTorch GPU版本..."
            echo "尝试多种安装方式..."
            
            # 方法1: 尝试使用PyTorch官方的最新稳定版
            echo -e "\n尝试方法1: PyTorch官方最新稳定版..."
            for i in $(seq 1 25); do
                sleep 0.1
                show_colored_progress $i 25 "安装PyTorch GPU版本"
            done
            if pip install torch torchvision torchaudio --no-cache-dir > /dev/null 2>&1; then
                echo -e "${GREEN}✅ 方法1成功${NC}"
            else
                echo -e "${YELLOW}方法1失败，尝试方法2...${NC}"
                
                # 方法2: 尝试使用CUDA 11.8版本
                echo "尝试方法2: CUDA 11.8版本..."
                for i in $(seq 1 25); do
                    sleep 0.1
                    show_colored_progress $i 25 "安装PyTorch CUDA 11.8"
                done
                if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir > /dev/null 2>&1; then
                    echo -e "${GREEN}✅ 方法2成功${NC}"
                else
                    echo -e "${YELLOW}方法2失败，尝试方法3...${NC}"
                    
                    # 方法3: 尝试使用CUDA 12.1版本
                    echo "尝试方法3: CUDA 12.1版本..."
                    for i in $(seq 1 25); do
                        sleep 0.1
                        show_colored_progress $i 25 "安装PyTorch CUDA 12.1"
                    done
                    if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir > /dev/null 2>&1; then
                        echo -e "${GREEN}✅ 方法3成功${NC}"
                    else
                        echo -e "${YELLOW}方法3失败，回退到CPU版本...${NC}"
                        for i in $(seq 1 20); do
                            sleep 0.1
                            show_colored_progress $i 20 "安装PyTorch CPU版本"
                        done
                        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir > /dev/null 2>&1
                    fi
                fi
            fi
            ;;
        "universal")
            echo "安装PyTorch CPU+GPU通用版本..."
            echo "将安装支持GPU的版本，即使没有GPU也能在CPU上运行..."
            
            # 首先尝试安装标准版本（通常包含CPU和GPU支持）
            for i in $(seq 1 30); do
                sleep 0.1
                show_colored_progress $i 30 "安装PyTorch通用版本"
            done
            if pip install torch torchvision torchaudio --no-cache-dir > /dev/null 2>&1; then
                echo -e "${GREEN}✅ 安装成功 - 通用版本${NC}"
                echo "此版本支持："
                echo "- 有GPU时自动使用GPU加速"
                echo "- 无GPU时自动回退到CPU运行"
            else
                echo -e "${YELLOW}标准版本安装失败，尝试其他方法...${NC}"
                
                # 根据系统是否有CUDA来选择
                if [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = true ]; then
                    echo "系统有CUDA，尝试安装GPU版本..."
                    for i in $(seq 1 25); do
                        sleep 0.1
                        show_colored_progress $i 25 "安装PyTorch GPU版本"
                    done
                    if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir > /dev/null 2>&1; then
                        echo -e "${GREEN}✅ GPU版本安装成功${NC}"
                    else
                        echo -e "${YELLOW}GPU版本安装失败，安装CPU版本...${NC}"
                        for i in $(seq 1 20); do
                            sleep 0.1
                            show_colored_progress $i 20 "安装PyTorch CPU版本"
                        done
                        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir > /dev/null 2>&1
                    fi
                else
                    echo "系统无CUDA，安装CPU版本..."
                    for i in $(seq 1 20); do
                        sleep 0.1
                        show_colored_progress $i 20 "安装PyTorch CPU版本"
                    done
                    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir > /dev/null 2>&1
                fi
            fi
            ;;
    esac
    
    # 检查PyTorch安装是否成功
    echo -n "检查PyTorch安装状态..."
    for i in $(seq 1 10); do
        sleep 0.05
        show_colored_progress $i 10 "检查PyTorch安装"
    done
    
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
    ((CURRENT_STEP++))
    step_progress $CURRENT_STEP $TOTAL_STEPS "安装AI库"
    
    echo -e "\n${YELLOW}安装AI库...${NC}"
    echo "安装Transformers和其他AI库..."
    
    # 逐个安装，以便更好地处理错误
    AI_PACKAGES=("transformers" "accelerate" "sentencepiece" "protobuf" "einops" "tiktoken" "gradio" "fastapi" "uvicorn")
    TOTAL_AI_PACKAGES=${#AI_PACKAGES[@]}
    CURRENT_AI_PACKAGE=0
    
    for package in "${AI_PACKAGES[@]}"; do
        ((CURRENT_AI_PACKAGE++))
        echo "安装 $package ($CURRENT_AI_PACKAGE/$TOTAL_AI_PACKAGES)..."
        for i in $(seq 1 10); do
            sleep 0.02
            show_colored_progress $i 10 "安装 $package"
        done
        pip install "$package" --no-cache-dir > /dev/null 2>&1 || echo -e "${YELLOW}⚠️  $package 安装失败，继续安装其他包...${NC}"
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
for i in $(seq 1 10); do
    sleep 0.05
    show_colored_progress $i 10 "检查PyTorch安装"
done

if ! python -c "import torch" 2>/dev/null; then
    echo -e "\n${RED}❌ PyTorch未正确安装${NC}"
    echo "正在尝试重新安装PyTorch..."
    
    echo "尝试安装最新版本..."
    for i in $(seq 1 20); do
        sleep 0.1
        show_colored_progress $i 20 "重新安装PyTorch"
    done
    pip install torch torchvision torchaudio --no-cache-dir > /dev/null 2>&1
    
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
for i in $(seq 1 10); do
    sleep 0.05
    show_colored_progress $i 10 "检查GPU状态"
done

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
    echo "8. 暂时不添加模型 (跳过模型下载)"
    
    read -p "请选择 [1-8] (默认: 8): " model_choice
    
    case ${model_choice:-8} in
        1)
            MODEL_NAME="Qwen/Qwen3-0.5B-Instruct"
            MODEL_SIZE="约0.5GB"
            ;;
        2)
            MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
            MODEL_SIZE="约1.2GB"
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
        8)
            MODEL_NAME=""
            MODEL_SIZE=""
            echo -e "${YELLOW}选择暂时不添加模型，跳过模型下载${NC}"
            echo -e "${YELLOW}可以在之后手动修改 model_config.json 文件添加模型${NC}"
            ;;
    esac
    
    # 保存模型配置
    if [ -n "$MODEL_NAME" ]; then
        cat > "$MODEL_CONFIG" << EOF
{
  "model": "${MODEL_NAME}",
  "model_size": "${MODEL_SIZE}",
  "selected_date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
    else
        # 创建空的模型配置
        cat > "$MODEL_CONFIG" << EOF
{
  "model": "",
  "model_size": "未选择模型",
  "selected_date": "$(date '+%Y-%m-%d %H:%M:%S')",
  "note": "请手动编辑此文件添加模型，例如：{\"model\": \"Qwen/Qwen3-0.5B-Instruct\", \"model_size\": \"约0.5GB\"}"
}
EOF
    fi
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
if [ -n "$MODEL_NAME" ]; then
    echo "选择模型: $MODEL_NAME ($MODEL_SIZE)"
else
    echo "模型: 未选择模型 (跳过下载)"
fi
print_separator

# 生成main.py文件
echo "生成main.py文件..."
for i in $(seq 1 10); do
    sleep 0.05
    show_colored_progress $i 10 "生成main.py"
done

cat > main.py << 'EOF'
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
EOF

chmod +x main.py
echo -e "${GREEN}✅ main.py 文件生成成功${NC}"

print_separator
echo -e "${GREEN}🚀 启动AI应用...${NC}"
echo "将在虚拟环境中运行 main.py"
print_separator

# 运行主程序
ARGS=""
if [ "$WEB_MODE" = true ]; then
    ARGS="$ARGS --web"
elif [ "$API_MODE" = true ]; then
    ARGS="$ARGS --api"
fi

echo -e "${CYAN}启动虚拟环境中的 main.py...${NC}"
python main.py