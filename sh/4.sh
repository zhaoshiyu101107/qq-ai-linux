#!/bin/bash
# AI项目启动脚本

# 虚拟环境路径（修改为你的实际路径）
VENV_PATH="venv"

# 项目路径（修改为你的实际路径）
PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🤖 AI项目启动"
echo "="*60

# 检查并创建虚拟环境
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "虚拟环境不存在: $VENV_PATH"
    read -p "是否创建虚拟环境？(y/n): " create_venv
    
    if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
        echo "正在创建虚拟环境..."
        
        # 检查Python3是否可用
        if command -v python3 &> /dev/null; then
            python3 -m venv "$VENV_PATH"
            
            if [ $? -eq 0 ]; then
                echo "✅ 虚拟环境创建成功: $VENV_PATH"
                
                # 激活虚拟环境以安装必要包
                source "$VENV_PATH/bin/activate"
                
                # 升级pip
                echo "升级pip..."
                pip install --upgrade pip
                
                # 安装基础依赖
                echo "安装基础依赖..."
                pip install numpy pandas
                
                echo "虚拟环境已准备就绪。"
            else
                echo "❌ 虚拟环境创建失败，请检查Python环境。"
                echo "可能需要安装python3-venv包:"
                echo "Ubuntu/Debian: sudo apt-get install python3-venv"
                echo "CentOS/RHEL: sudo yum install python3-venv"
                echo "macOS: 确保已安装Python3"
                exit 1
            fi
        else
            echo "❌ 未找到python3，请先安装Python3。"
            echo "访问 https://www.python.org/downloads/ 获取安装包"
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

# 检查并确保安装GPU版本PyTorch
echo "检查PyTorch版本..."
if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "+cpu"; then
    echo "❌ 检测到CPU版本PyTorch，正在重新安装GPU版本..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null
    echo "安装GPU版本PyTorch（CUDA 11.8）..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "⚠️  PyTorch GPU版本已安装，但CUDA不可用，可能是驱动问题。"
fi

# 检查GPU
echo "检查GPU状态..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 GPU加速已启用 ({torch.cuda.device_count()}个GPU)')
else:
    print('⚠️  使用CPU运行，速度较慢')
"

# 进入项目目录
cd "$PROJECT_PATH"

echo "="*60
echo "项目目录: $PROJECT_PATH"
echo "虚拟环境: $VENV_PATH"
echo "="*60
echo "🚀 启动AI应用..."
echo "输入 'quit' 退出应用"
echo "="*60

# 运行主程序
python main.py