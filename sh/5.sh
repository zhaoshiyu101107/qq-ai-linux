#!/bin/bash
# AI项目启动脚本

# 虚拟环境路径
VENV_PATH="venv"

# 项目路径
PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 打印分隔线
print_separator() {
    printf '=%.0s' {1..60}
    echo ""
}

# 进度条函数
show_progress() {
    local width=50
    local percent=$1
    local filled=$((width * percent / 100))
    local empty=$((width - filled))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %3d%%" $percent
}

echo "🤖 AI项目启动"
print_separator

# 检查并创建虚拟环境
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "虚拟环境不存在: $VENV_PATH"
    read -p "是否创建虚拟环境？(y/n): " create_venv
    
    if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
        echo "正在创建虚拟环境..."
        
        # 检查Python3是否可用
        if command -v python3 &> /dev/null; then
            # 创建虚拟环境
            echo -n "创建虚拟环境..."
            python3 -m venv "$VENV_PATH" 2>/dev/null &
            python_pid=$!
            
            # 显示简单进度条
            for i in {1..20}; do
                show_progress $((i*5))
                sleep 0.2
                if ! kill -0 $python_pid 2>/dev/null; then
                    break
                fi
            done
            show_progress 100
            echo ""
            
            wait $python_pid
            
            if [ $? -eq 0 ]; then
                echo "✅ 虚拟环境创建成功: $VENV_PATH"
                
                # 激活虚拟环境
                source "$VENV_PATH/bin/activate"
                
                # 升级pip（显示进度）
                echo -n "升级pip..."
                pip install --upgrade pip -q 2>/dev/null &
                pip_pid=$!
                
                for i in {1..20}; do
                    show_progress $((i*5))
                    sleep 0.1
                    if ! kill -0 $pip_pid 2>/dev/null; then
                        break
                    fi
                done
                show_progress 100
                echo ""
                wait $pip_pid
                
                # 安装基础依赖（显示进度）
                echo -n "安装基础依赖..."
                {
                pip install numpy -q 2>/dev/null
                pip install pandas -q 2>/dev/null
                } &
                deps_pid=$!
                
                for i in {1..20}; do
                    show_progress $((i*5))
                    sleep 0.2
                    if ! kill -0 $deps_pid 2>/dev/null; then
                        break
                    fi
                done
                show_progress 100
                echo ""
                wait $deps_pid
                
                echo "✅ 基础依赖安装完成"
            else
                echo ""
                echo "❌ 虚拟环境创建失败"
                exit 1
            fi
        else
            echo "❌ 未找到python3，请先安装Python3。"
            exit 1
        fi
    else
        echo "❌ 用户取消创建虚拟环境，退出程序。"
        exit 1
    fi
else
    # 激活虚拟环境
    echo "激活虚拟环境..."
    source "$VENV_PATH/bin/activate"
fi

# 检查PyTorch是否已安装
echo -n "检查PyTorch安装状态..."
if ! python -c "import torch" 2>/dev/null; then
    echo ""
    echo "❌ 未检测到PyTorch，正在安装PyTorch..."
    
    # 检测CUDA是否可用
    if command -v nvidia-smi &> /dev/null; then
        echo "检测到NVIDIA GPU，安装GPU版本PyTorch（CUDA 11.8）..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir 2>&1 | \
        while read -r line; do
            if [[ $line == *"Installing"* ]] || [[ $line == *"Downloading"* ]]; then
                echo -ne "\r$line"
            fi
        done
        echo ""
        echo "✅ PyTorch GPU版本安装完成"
    else
        echo "未检测到NVIDIA GPU，安装CPU版本PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir 2>&1 | \
        while read -r line; do
            if [[ $line == *"Installing"* ]] || [[ $line == *"Downloading"* ]]; then
                echo -ne "\r$line"
            fi
        done
        echo ""
        echo "✅ PyTorch CPU版本安装完成"
    fi
else
    echo "✅ PyTorch已安装"
    
    # 检查是否是CPU版本
    if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "+cpu"; then
        echo "检测到CPU版本PyTorch"
        if command -v nvidia-smi &> /dev/null; then
            echo "检测到NVIDIA GPU，建议升级到GPU版本以获得更好的性能"
            read -p "是否升级到GPU版本？(y/n): " upgrade_pytorch
            if [[ $upgrade_pytorch == "y" || $upgrade_pytorch == "Y" ]]; then
                echo "正在重新安装GPU版本PyTorch（CUDA 11.8）..."
                pip uninstall -y torch torchvision torchaudio 2>/dev/null
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
                echo "✅ PyTorch GPU版本安装完成"
            fi
        fi
    fi
fi

# 检查GPU
echo "检查GPU状态..."
if python -c "import torch" 2>/dev/null; then
    python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 GPU加速已启用 ({torch.cuda.device_count()}个GPU)')
else:
    if command -v nvidia-smi &> /dev/null:
        print('⚠️  检测到NVIDIA GPU，但CUDA不可用，请检查驱动和CUDA安装')
    else:
        print('⚠️  未检测到GPU，使用CPU运行，速度较慢')
"
else
    echo "❌ PyTorch未正确安装"
    exit 1
fi

# 进入项目目录
cd "$PROJECT_PATH"

print_separator
echo "项目目录: $PROJECT_PATH"
echo "虚拟环境: $VENV_PATH"
print_separator
echo "🚀 启动AI应用..."
echo "输入 'quit' 退出应用"
print_separator

# 运行主程序
python main.py