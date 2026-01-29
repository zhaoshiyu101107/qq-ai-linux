#!/bin/bash
# 自包含AI环境安装脚本 - 所有内容安装到脚本所在目录

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🚀 开始创建自包含AI环境...${NC}"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_HOME="${SCRIPT_DIR}/ai_env"

echo "安装目录: ${AI_HOME}"

# 创建目录结构
mkdir -p "${AI_HOME}"
cd "${AI_HOME}"

# 创建虚拟环境（隔离环境）
echo -e "\n${YELLOW}[1/5] 创建Python虚拟环境...${NC}"
python3 -m venv venv 2>/dev/null || python -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
echo -e "\n${YELLOW}[2/5] 升级pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1

# 检测GPU
echo -e "\n${YELLOW}[3/5] 检测系统环境...${NC}"

# 检测NVIDIA GPU
HAS_CUDA=false
HAS_NVIDIA=false
PYTORCH_VERSION="cpu"

# 检查nvidia-smi命令
if command -v nvidia-smi &> /dev/null; then
    HAS_NVIDIA=true
    echo -e "${GREEN}✓ 检测到NVIDIA GPU${NC}"
    
    # 检查CUDA版本
    if nvidia-smi | grep -q "CUDA Version"; then
        HAS_CUDA=true
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
        echo -e "${GREEN}✓ 检测到CUDA版本: ${CUDA_VERSION}.x${NC}"
    else
        echo -e "${YELLOW}⚠ NVIDIA驱动已安装但未检测到CUDA${NC}"
    fi
else
    echo -e "${YELLOW}ℹ 未检测到NVIDIA GPU或驱动${NC}"
    
    # 检查是否有其他GPU
    if command -v lspci &> /dev/null; then
        if lspci | grep -i "vga\|3d\|display" | grep -v "NVIDIA"; then
            echo -e "${YELLOW}⚠ 检测到其他显卡（AMD/Intel），仅支持CPU模式${NC}"
        fi
    fi
fi

# 询问用户选择PyTorch版本
echo -e "\n${BLUE}请选择PyTorch安装版本:${NC}"
if [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = true ]; then
    echo "1. GPU加速版 (CUDA ${CUDA_VERSION}.x) - 推荐，需要NVIDIA GPU"
    echo "2. CPU版 - 通用兼容，无GPU加速"
    echo "3. CPU+GPU版 - 同时安装CPU和GPU支持"
    
    read -p "请选择 [1/2/3] (默认: 1): " choice
    
    case $choice in
        2)
            PYTORCH_VERSION="cpu"
            echo -e "${YELLOW}选择: CPU版本${NC}"
            ;;
        3)
            PYTORCH_VERSION="both"
            echo -e "${YELLOW}选择: CPU+GPU版本${NC}"
            ;;
        *)
            PYTORCH_VERSION="gpu"
            echo -e "${YELLOW}选择: GPU加速版本${NC}"
            ;;
    esac
elif [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = false ]; then
    echo "1. CPU版 - NVIDIA驱动已安装但CUDA不可用"
    echo "2. 尝试安装GPU版 - 可能需要额外配置CUDA"
    
    read -p "请选择 [1/2] (默认: 1): " choice
    
    if [ "$choice" = "2" ]; then
        PYTORCH_VERSION="gpu"
        echo -e "${YELLOW}选择: 尝试安装GPU版本${NC}"
        echo -e "${YELLOW}注意: 如果CUDA未正确安装，可能需要手动安装CUDA工具包${NC}"
    else
        PYTORCH_VERSION="cpu"
        echo -e "${YELLOW}选择: CPU版本${NC}"
    fi
else
    echo "1. CPU版 - 唯一可用选项"
    PYTORCH_VERSION="cpu"
    echo -e "${YELLOW}选择: CPU版本${NC}"
fi

# 根据选择安装PyTorch
echo -e "\n${YELLOW}[4/5] 安装PyTorch...${NC}"

case $PYTORCH_VERSION in
    "cpu")
        echo "安装PyTorch CPU版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
        ;;
    "gpu")
        echo "安装PyTorch GPU版本..."
        
        # 根据检测到的CUDA版本选择对应的PyTorch
        if [ "$CUDA_VERSION" = "12" ] || [ "$CUDA_VERSION" = "12.1" ] || [ "$CUDA_VERSION" = "12.2" ] || [ "$CUDA_VERSION" = "12.3" ] || [ "$CUDA_VERSION" = "12.4" ]; then
            echo "使用CUDA 12.1版本..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 > /dev/null 2>&1
        elif [ "$CUDA_VERSION" = "11" ] || [ "$CUDA_VERSION" = "11.8" ]; then
            echo "使用CUDA 11.8版本..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
        else
            echo "使用默认CUDA版本 (11.8)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
        fi
        ;;
    "both")
        echo "安装PyTorch CPU+GPU版本..."
        echo "注意: 这将安装较大的包，同时支持CPU和GPU运行"
        pip install torch torchvision torchaudio > /dev/null 2>&1
        ;;
esac

# 安装Transformers和其他AI库
echo "安装Transformers和其他AI库..."
pip install transformers accelerate sentencepiece protobuf einops tiktoken > /dev/null 2>&1

# 创建配置文件
echo -e "\n${YELLOW}[5/5] 创建配置文件...${NC}"

# 获取当前日期
CURRENT_DATE=$(date '+%Y-%m-%d %H:%M:%S')

cat > config.json << EOF
{
  "environment": "local",
  "model": "Qwen/Qwen3-0.6B-Instruct",
  "install_date": "${CURRENT_DATE}",
  "install_dir": "${AI_HOME}",
  "pytorch_version": "${PYTORCH_VERSION}",
  "has_gpu": ${HAS_NVIDIA},
  "has_cuda": ${HAS_CUDA},
  "cuda_version": "${CUDA_VERSION:-null}",
  "requirements": [
    "torch",
    "transformers",
    "accelerate",
    "sentencepiece",
    "protobuf",
    "einops",
    "tiktoken"
  ]
}
EOF

# 创建启动脚本
echo "创建启动脚本..."

cat > start_qwen.sh << 'EOF'
#!/bin/bash
# Qwen3-0.6B 启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🤖 Qwen3-0.6B 本地运行环境"
echo "=========================================="
echo "位置: $SCRIPT_DIR"
echo ""

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "激活虚拟环境..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "错误: 未找到虚拟环境"
        exit 1
    fi
fi

# 运行Python脚本
python run_qwen.py "$@"
EOF

chmod +x start_qwen.sh

# 创建主运行脚本
cat > run_qwen.py << 'EOF'
#!/usr/bin/env python3
"""
Qwen3-0.6B 本地运行脚本
无需网络连接，完全本地运行
"""

import os
import sys
import torch
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def check_gpu_info():
    """检查GPU信息"""
    print("=" * 50)
    print("🤖 Qwen3-0.6B - 本地AI环境")
    print("=" * 50)
    print(f"目录: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ 使用CPU模式 - 速度较慢")
        print("💡 提示: 如需GPU加速，请确保已安装NVIDIA驱动和CUDA")
    
    print("=" * 50)
    print()

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    check_gpu_info()
    
    print("正在加载Qwen3-0.6B模型...")
    print("首次运行需要下载模型文件（约1.2GB）")
    print("下载完成后会缓存，下次无需重新下载")
    print()
    
    # 模型名称
    model_name = "Qwen/Qwen3-0.6B-Instruct"
    
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
        
        # 根据是否有GPU选择加载方式
        if torch.cuda.is_available():
            print("  使用GPU加速")
            try:
                # 尝试使用GPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"  GPU加载失败: {e}")
                print("  回退到CPU模式...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
        else:
            print("  使用CPU运行")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        
        print()
        print("✅ 模型加载成功！")
        print("-" * 40)
        print("💡 提示: 输入 'quit' 或 'exit' 退出")
        print("-" * 40)
        
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
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
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
                if "assistant" in response:
                    response = response.split("assistant")[-1].strip()
                elif "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                
                print(f"Qwen: {response}")
                
            except KeyboardInterrupt:
                print("\n\n退出程序")
                break
            except Exception as e:
                print(f"\n错误: {e}")
                continue
                
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 磁盘空间不足")
        print("3. 内存不足")
        print("\n💡 解决方案:")
        print(f"   检查目录: {cache_dir}")
        print("   确保有至少2GB可用空间")
        print("   检查网络连接")

if __name__ == "__main__":
    main()
EOF

# 创建工具脚本
cat > tools.py << 'EOF'
#!/usr/bin/env python3
"""
AI环境工具脚本
"""

import torch
import sys
import os
import json

def check_environment():
    """检查环境状态"""
    print("🔍 环境检查")
    print("-" * 50)
    
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"安装日期: {config.get('install_date', '未知')}")
        print(f"PyTorch版本: {config.get('pytorch_version', '未知')}")
        print(f"检测到GPU: {config.get('has_gpu', False)}")
        if config.get('has_cuda'):
            print(f"CUDA版本: {config.get('cuda_version', '未知')}")
    else:
        print("配置文件未找到")
    
    print("-" * 50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠ 当前运行在CPU模式")
    
    # 检查其他包
    packages = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("einops", "einops"),
    ]
    
    print("-" * 50)
    for name, module in packages:
        try:
            __import__(module)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
    
    print("-" * 50)

def clear_cache():
    """清理模型缓存"""
    import shutil
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    
    if os.path.exists(cache_dir):
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        size_gb = total_size / (1024**3)
        
        print(f"缓存文件: {file_count} 个")
        print(f"缓存大小: {size_gb:.2f} GB")
        print()
        
        response = input("确认删除所有缓存文件？(y/N): ").strip().lower()
        
        if response == 'y':
            shutil.rmtree(cache_dir)
            print("✅ 缓存已清理")
        else:
            print("❌ 取消操作")
    else:
        print("✅ 缓存目录不存在")

def switch_mode():
    """切换运行模式"""
    print("🔄 切换运行模式")
    print("-" * 50)
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        print("❌ 配置文件不存在")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    current_mode = config.get('pytorch_version', 'cpu')
    print(f"当前模式: {current_mode}")
    print()
    print("可用模式:")
    print("1. CPU模式 - 兼容性好，速度慢")
    print("2. GPU模式 - 需要NVIDIA GPU和CUDA")
    print("3. 双模式 - 同时支持CPU和GPU")
    
    choice = input("\n选择模式 [1/2/3] (按Enter取消): ").strip()
    
    if choice == "1":
        new_mode = "cpu"
    elif choice == "2":
        new_mode = "gpu"
    elif choice == "3":
        new_mode = "both"
    else:
        print("❌ 取消操作")
        return
    
    if new_mode != current_mode:
        config['pytorch_version'] = new_mode
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ 已切换到 {new_mode} 模式")
        print("💡 提示: 需要重新安装PyTorch才能使更改生效")
    else:
        print("ℹ 模式未改变")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_environment()
        elif sys.argv[1] == "clear":
            clear_cache()
        elif sys.argv[1] == "switch":
            switch_mode()
        else:
            print("用法: python tools.py [check|clear|switch]")
            print("  check  - 检查环境状态")
            print("  clear  - 清理模型缓存")
            print("  switch - 切换运行模式")
    else:
        check_environment()
EOF
