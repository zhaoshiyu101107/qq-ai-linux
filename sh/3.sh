#!/bin/bash
# 自包含AI环境安装脚本 - 支持GPU检测和模型选择

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
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
echo -e "\n${YELLOW}[1/7] 创建Python虚拟环境...${NC}"
python3 -m venv venv 2>/dev/null || python -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
echo -e "\n${YELLOW}[2/7] 升级pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1

# 检测GPU
echo -e "\n${YELLOW}[3/7] 检测系统环境...${NC}"

# 检测NVIDIA GPU
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
    
    # 检查是否有其他GPU
    if command -v lspci &> /dev/null; then
        if lspci | grep -i "vga\|3d\|display" | grep -qi "nvidia"; then
            echo -e "${YELLOW}⚠ 检测到NVIDIA显卡但驱动未安装${NC}"
            HAS_NVIDIA=true
        fi
    fi
fi

# 询问用户选择PyTorch版本
echo -e "\n${BLUE}请选择PyTorch安装版本:${NC}"

if [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = true ]; then
    echo "1. GPU加速版 (CUDA ${CUDA_VERSION}.x) - 推荐，需要NVIDIA GPU"
    echo "2. CPU版 - 通用兼容，无GPU加速"
    echo "3. CPU+GPU版 - 同时安装CPU和GPU支持"
    echo "4. 自动选择 - 根据系统自动选择最佳版本"
    
    read -p "请选择 [1/2/3/4] (默认: 4): " choice
    
    case $choice in
        1)
            PYTORCH_VERSION="gpu"
            echo -e "${YELLOW}选择: GPU加速版本${NC}"
            ;;
        2)
            PYTORCH_VERSION="cpu"
            echo -e "${YELLOW}选择: CPU版本${NC}"
            ;;
        3)
            PYTORCH_VERSION="both"
            echo -e "${YELLOW}选择: CPU+GPU版本${NC}"
            ;;
        4|"")
            PYTORCH_VERSION="auto"
            echo -e "${YELLOW}选择: 自动选择版本${NC}"
            ;;
        *)
            PYTORCH_VERSION="auto"
            echo -e "${YELLOW}选择: 自动选择版本${NC}"
            ;;
    esac
elif [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = false ]; then
    echo "1. CPU版 - NVIDIA驱动已安装但CUDA可能不可用"
    echo "2. 尝试安装GPU版 - 可能需要额外配置CUDA"
    echo "3. 自动选择 - 根据系统自动选择"
    
    read -p "请选择 [1/2/3] (默认: 3): " choice
    
    case $choice in
        1)
            PYTORCH_VERSION="cpu"
            echo -e "${YELLOW}选择: CPU版本${NC}"
            ;;
        2)
            PYTORCH_VERSION="gpu"
            echo -e "${YELLOW}选择: 尝试安装GPU版本${NC}"
            echo -e "${YELLOW}注意: 如果CUDA未正确安装，可能需要手动安装CUDA工具包${NC}"
            ;;
        3|"")
            PYTORCH_VERSION="auto"
            echo -e "${YELLOW}选择: 自动选择版本${NC}"
            ;;
        *)
            PYTORCH_VERSION="auto"
            echo -e "${YELLOW}选择: 自动选择版本${NC}"
            ;;
    esac
else
    echo "1. CPU版 - 唯一可用选项"
    echo "2. 自动选择 - 根据系统自动选择"
    read -p "请选择 [1/2] (默认: 2): " choice
    
    if [ "$choice" = "1" ]; then
        PYTORCH_VERSION="cpu"
        echo -e "${YELLOW}选择: CPU版本${NC}"
    else
        PYTORCH_VERSION="auto"
        echo -e "${YELLOW}选择: 自动选择版本${NC}"
    fi
fi

# 根据选择安装PyTorch
echo -e "\n${YELLOW}[4/7] 安装PyTorch...${NC}"

if [ "$PYTORCH_VERSION" = "auto" ]; then
    if [ "$HAS_NVIDIA" = true ] && [ "$HAS_CUDA" = true ]; then
        echo "自动选择: 安装GPU版本 (CUDA ${CUDA_VERSION}.x)"
        PYTORCH_VERSION="gpu"
    else
        echo "自动选择: 安装CPU版本"
        PYTORCH_VERSION="cpu"
    fi
fi

case $PYTORCH_VERSION in
    "cpu")
        echo "安装PyTorch CPU版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
        ;;
    "gpu")
        echo "安装PyTorch GPU版本..."
        
        # 根据检测到的CUDA版本选择对应的PyTorch
        if [ -n "$CUDA_VERSION" ]; then
            if [ "$CUDA_VERSION" = "12" ] || [ "$CUDA_VERSION" -ge 12 ]; then
                echo "使用CUDA 12.1版本..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 > /dev/null 2>&1
            elif [ "$CUDA_VERSION" = "11" ]; then
                echo "使用CUDA 11.8版本..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
            else
                echo "使用默认CUDA版本 (11.8)..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
            fi
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
echo -e "\n${YELLOW}[5/7] 安装Transformers和其他AI库...${NC}"
echo "安装Transformers..."
pip install transformers accelerate sentencepiece protobuf einops tiktoken > /dev/null 2>&1

# 询问用户选择模型
echo -e "\n${YELLOW}[6/7] 选择AI模型...${NC}"
echo -e "${BLUE}请选择要使用的AI模型:${NC}"
echo "1. Qwen/Qwen3-0.5B-Instruct (轻量级, 约0.5GB)"
echo "2. Qwen/Qwen3-0.6B-Instruct (推荐, 约1.2GB)"
echo "3. Qwen/Qwen3-1.8B-Instruct (平衡, 约3.6GB)"
echo "4. Qwen/Qwen3-4B-Instruct (性能好, 约8GB)"
echo "5. Qwen/Qwen2.5-0.5B-Instruct (新版, 约0.5GB)"
echo "6. microsoft/phi-2 (微软Phi-2, 约2.7GB)"
echo "7. TinyLlama/TinyLlama-1.1B-Chat-v1.0 (小羊驼, 约2.2GB)"
echo "8. 自定义模型 (输入完整的HuggingFace模型路径)"

read -p "请选择 [1-8] (默认: 2): " model_choice

case $model_choice in
    1)
        MODEL_NAME="Qwen/Qwen3-0.5B-Instruct"
        MODEL_SIZE="约0.5GB"
        ;;
    3)
        MODEL_NAME="Qwen/Qwen3-1.8B-Instruct"
        MODEL_SIZE="约3.6GB"
        ;;
    4)
        MODEL_NAME="Qwen/Qwen3-4B-Instruct"
        MODEL_SIZE="约8GB"
        ;;
    5)
        MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
        MODEL_SIZE="约0.5GB"
        ;;
    6)
        MODEL_NAME="microsoft/phi-2"
        MODEL_SIZE="约2.7GB"
        ;;
    7)
        MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        MODEL_SIZE="约2.2GB"
        ;;
    8)
        read -p "请输入完整的HuggingFace模型路径 (例如: Qwen/Qwen3-0.6B-Instruct): " custom_model
        if [ -n "$custom_model" ]; then
            MODEL_NAME="$custom_model"
            MODEL_SIZE="未知大小"
            echo -e "${YELLOW}注意: 使用自定义模型: ${MODEL_NAME}${NC}"
        else
            MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
            MODEL_SIZE="约1.2GB"
            echo -e "${YELLOW}使用默认模型: ${MODEL_NAME}${NC}"
        fi
        ;;
    2|"")
        MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
        MODEL_SIZE="约1.2GB"
        ;;
    *)
        MODEL_NAME="Qwen/Qwen3-0.6B-Instruct"
        MODEL_SIZE="约1.2GB"
        echo -e "${YELLOW}使用默认模型: ${MODEL_NAME}${NC}"
        ;;
esac

# 询问是否安装附加组件
echo -e "\n${BLUE}是否安装额外的AI组件？${NC}"
echo "1. 基础组件 (已安装)"
echo "2. 添加LangChain支持 (AI应用开发)"
echo "3. 添加Gradio Web界面"
echo "4. 添加Jupyter支持 (交互式编程)"
echo "5. 全部安装"

read -p "请选择 [1-5] (默认: 1): " extra_choice

case $extra_choice in
    2)
        echo "安装LangChain..."
        pip install langchain langchain-community > /dev/null 2>&1
        EXTRA_PACKAGES="langchain"
        ;;
    3)
        echo "安装Gradio..."
        pip install gradio > /dev/null 2>&1
        EXTRA_PACKAGES="gradio"
        ;;
    4)
        echo "安装Jupyter..."
        pip install jupyter ipykernel > /dev/null 2>&1
        EXTRA_PACKAGES="jupyter"
        ;;
    5)
        echo "安装所有额外组件..."
        pip install langchain langchain-community gradio jupyter ipykernel > /dev/null 2>&1
        EXTRA_PACKAGES="all"
        ;;
    *)
        echo "仅安装基础组件"
        EXTRA_PACKAGES="none"
        ;;
esac

# 创建配置文件
echo -e "\n${YELLOW}[7/7] 创建配置文件...${NC}"

# 获取当前日期
CURRENT_DATE=$(date '+%Y-%m-%d %H:%M:%S')

# 创建模型配置目录
mkdir -p model_configs

cat > config.json << EOF
{
  "environment": "local",
  "model": "${MODEL_NAME}",
  "model_size": "${MODEL_SIZE}",
  "install_date": "${CURRENT_DATE}",
  "install_dir": "${AI_HOME}",
  "pytorch_version": "${PYTORCH_VERSION}",
  "has_gpu": ${HAS_NVIDIA},
  "has_cuda": ${HAS_CUDA},
  "cuda_version": "${CUDA_VERSION:-null}",
  "extra_packages": "${EXTRA_PACKAGES}",
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

# 为选择的模型创建特定配置
MODEL_SHORT_NAME=$(echo "$MODEL_NAME" | sed 's/[\/-]/_/g')
cat > "model_configs/${MODEL_SHORT_NAME}.json" << EOF
{
  "model_name": "${MODEL_NAME}",
  "model_type": "causal_lm",
  "tokenizer_class": "AutoTokenizer",
  "model_class": "AutoModelForCausalLM",
  "trust_remote_code": true,
  "features": {
    "chat_template": true,
    "generation": true,
    "streaming": false
  },
  "recommended_settings": {
    "max_length": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
  }
}
EOF

# 创建启动脚本
echo "创建启动脚本..."

cat > start_ai.sh << EOF
#!/bin/bash
# AI模型启动脚本

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
cd "\$SCRIPT_DIR"

echo "=========================================="
echo "🤖 本地AI运行环境"
echo "=========================================="
echo "模型: ${MODEL_NAME}"
echo "位置: \$SCRIPT_DIR"
echo ""

# 检查是否在虚拟环境中
if [ -z "\$VIRTUAL_ENV" ]; then
    echo "激活虚拟环境..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "错误: 未找到虚拟环境"
        exit 1
    fi
fi

# 运行Python脚本
python run_ai.py "\$@"
EOF

chmod +x start_ai.sh

# 创建模型选择脚本
cat > switch_model.sh << 'EOF'
#!/bin/bash
# 切换AI模型脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
fi

python switch_model.py "$@"
EOF

chmod +x switch_model.sh

# 创建主运行脚本
cat > run_ai.py << 'EOF'
#!/usr/bin/env python3
"""
AI模型本地运行脚本
支持多种模型，完全本地运行
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
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
    print(f"模型: {config.get('model', '未知')}")
    print(f"模型大小: {config.get('model_size', '未知')}")
    print(f"Python: {sys.version.split()[0]}")
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
        if config.get('has_gpu'):
            print("💡 提示: 检测到GPU但PyTorch无法访问")
            print("💡 可能原因: PyTorch安装的是CPU版本")
            print("💡 解决方案: 重新安装GPU版本的PyTorch")
    
    print("=" * 50)
    print()

def load_model():
    """加载AI模型"""
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
        
        return tokenizer, model
        
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        return None, None

def chat_loop(tokenizer, model):
    """对话循环"""
    print()
    print("✅ 模型加载成功！")
    print("-" * 50)
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
        except Exception as e:
            print(f"\n错误: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="运行本地AI模型")
    parser.add_argument("--model", type=str, help="指定要使用的模型")
    parser.add_argument("--check", action="store_true", help="仅检查环境")
    parser.add_argument("--web", action="store_true", help="启动Web界面")
    parser.add_argument("--api", action="store_true", help="启动API服务")
    
    args = parser.parse_args()
    
    if args.check:
        check_gpu_info()
        return
    
    check_gpu_info()
    
    if args.model:
        # 切换到指定模型
        config = load_config()
        config['model'] = args.model
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"已切换到模型: {args.model}")
    
    # 加载模型
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        print("\n💡 可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 检查模型名称是否正确")
        print(f"4. 尝试其他模型: python switch_model.py")
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

# 创建模型切换脚本
cat > switch_model.py << 'EOF'
#!/usr/bin/env python3
"""
AI模型切换脚本
"""

import os
import sys
import json

def main():
    print("🤖 AI模型切换工具")
    print("=" * 50)
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        print("❌ 配置文件不存在")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    current_model = config.get('model', '未知')
    print(f"当前模型: {current_model}")
    print()
    
    # 可用模型列表
    available_models = {
        "1": ("Qwen/Qwen3-0.5B-Instruct", "轻量级, 约0.5GB"),
        "2": ("Qwen/Qwen3-0.6B-Instruct", "推荐, 约1.2GB"),
        "3": ("Qwen/Qwen3-1.8B-Instruct", "平衡, 约3.6GB"),
        "4": ("Qwen/Qwen3-4B-Instruct", "性能好, 约8GB"),
        "5": ("Qwen/Qwen2.5-0.5B-Instruct", "新版, 约0.5GB"),
        "6": ("microsoft/phi-2", "微软Phi-2, 约2.7GB"),
        "7": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "小羊驼, 约2.2GB"),
        "8": ("自定义", "输入完整的HuggingFace模型路径")
    }
    
    print("可用模型:")
    for key, (name, desc) in available_models.items():
        print(f"{key}. {name} - {desc}")
    
    print()
    choice = input("选择模型编号 [1-8] (按Enter取消): ").strip()
    
    if not choice:
        print("❌ 取消操作")
        return
    
    if choice == "8":
        custom_model = input("请输入完整的HuggingFace模型路径: ").strip()
        if custom_model:
            new_model = custom_model
            model_size = "未知大小"
        else:
            print("❌ 未输入模型路径")
            return
    elif choice in available_models:
        new_model, model_desc = available_models[choice]
        model_size = model_desc.split(",")[-1].strip()
    else:
        print("❌ 无效选择")
        return
    
    if new_model == current_model:
        print(f"ℹ 模型未改变: {current_model}")
        return
    
    # 更新配置
    config['model'] = new_model
    config['model_size'] = model_size
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 已切换到模型: {new_model}")
    print(f"📊 模型大小: {model_size}")
    print()
    print("💡 下次启动时将使用新模型")
    print("💡 首次使用新模型需要下载模型文件")

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
    print("=" * 50)
    
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"安装日期: {config.get('install_date', '未知')}")
        print(f"当前模型: {config.get('model', '未知')}")
        print(f"模型大小: {config.get('model_size', '未知')}")
        print(f"PyTorch版本: {config.get('pytorch_version', '未知')}")
        print(f"检测到GPU: {config.get('has_gpu', False)}")
        if config.get('has_cuda'):
            print(f"CUDA版本: {config.get('cuda_version', '未知')}")
    else:
        print("配置文件未找到")
    
    print("=" * 50)
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
        if config.get('has_gpu'):
            print("💡 检测到GPU但PyTorch无法访问")
            print("💡 可能安装了CPU版本的PyTorch")
            print("💡 建议重新安装GPU版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # 检查其他包
    packages = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("einops", "einops"),
    ]
    
    print("=" * 50)
    for name, module in packages:
        try:
            __import__(module)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
    
    print("=" * 50)

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

def diagnose_gpu():
    """诊断GPU问题"""
    print("🔧 GPU诊断工具")
    print("=" * 50)
    
    print("PyTorch信息:")
    print(f"  版本: {torch.__version__}")
    print(f"  CUDA编译版本: {torch.version.cuda}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    print("\n系统信息:")
    import platform
    print(f"  操作系统: {platform.system()} {platform.release()}")
    
    # 检查nvidia-smi
    print("\n检查NVIDIA驱动:")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi命令可用")
            lines = result.stdout.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("❌ nvidia-smi命令不可用")
    except FileNotFoundError:
        print("❌ 未找到nvidia-smi命令")
    
    print("\n环境变量检查:")
    env_vars = ['CUDA_HOME', 'CUDA_PATH', 'PATH']
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        if var == 'PATH':
            print(f"  {var}: (长度: {len(value)} 字符)")
        else:
            print(f"  {var}: {value}")
    
    print("\n💡 建议:")
    if not torch.cuda.is_available():
        print("1. 确保已安装NVIDIA驱动")
        print("2. 安装CUDA工具包")
        print("3. 重新安装GPU版本的PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_environment()
        elif sys.argv[1] == "clear":
            clear_cache()
        elif sys.argv[1] == "diagnose":
            diagnose_gpu()
        else:
            print("用法: python tools.py [check|clear|diagnose]")
            print("  check    - 检查环境状态")
            print("  clear    - 清理模型缓存")
            print("  diagnose - 诊断GPU问题")
    else:
        check_environment()
EOF

# 创建修复脚本专门针对GPU问题
cat > fix_gpu.sh << 'EOF'
#!/bin/bash
# 修复GPU支持脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 修复GPU支持"
echo "================"

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "❌ 未找到虚拟环境"
        exit 1
    fi
fi

# 检查当前PyTorch版本
echo "当前PyTorch版本:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

echo -e "\n是否重新安装GPU版本的PyTorch？ (y/N)"
read -p "选择: " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo "正在卸载当前PyTorch..."
    pip uninstall torch torchvision torchaudio -y
    
    echo -e "\n选择CUDA版本:"
    echo "1. CUDA 11.8 (兼容性好)"
    echo "2. CUDA 12.1 (较新版本)"
    read -p "选择 [1/2] (默认: 1): " cuda_choice
    
    if [ "$cuda_choice" = "2" ]; then
        echo "安装CUDA 12.1版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "安装CUDA 11.8版本..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    
    echo -e "\n✅ 安装完成！"
    echo "新的PyTorch版本:"
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
else
    echo "❌ 取消操作"
fi
EOF

chmod +x fix_gpu.sh

# 创建README文件
cat > README.md << EOF
# 🤖 本地AI环境

这是一个完全自包含的AI运行环境，支持多种AI模型和GPU加速。

## 📦 安装信息

- **安装目录**: ${AI_HOME}
- **选择模型**: ${MODEL_NAME} (${MODEL_SIZE})
- **PyTorch版本**: ${PYTORCH_VERSION}
- **GPU支持**: ${HAS_NVIDIA}
- **CUDA版本**: ${CUDA_VERSION:-未检测到}
- **安装时间**: ${CURRENT_DATE}

## 🚀 快速开始

1. **启动AI聊天**:
   \`\`\`bash
   ./start_ai.sh
   \`\`\`

2. **启动Web界面** (需要安装gradio):
   \`\`\`bash
   ./start_ai.sh --web
   \`\`\`

3. **检查环境**:
   \`\`\`bash
   python tools.py check
   \`\`\`

## 🛠️ 工具命令

- \`./start_ai.sh\` - 启动AI聊天
- \`./start_ai.sh --web\` - 启动Web界面
- \`./start_ai.sh --api\` - 启动API服务
- \`python tools.py check\` - 检查环境状态
- \`python tools.py diagnose\` - 诊断GPU问题
- \`python tools.py clear\` - 清理模型缓存
- \`python switch_model.py\` - 切换AI模型
- \`./fix_gpu.sh\` - 修复GPU支持

## 🔧 解决GPU检测问题

如果你的系统有GPU但PyTorch检测不到，请运行:

\`\`\`bash
./fix_gpu.sh
\`\`\`

或者手动重新安装GPU版本的PyTorch:

\`\`\`bash
source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
\`\`\`

## 📁 文件说明

- \`config.json\` - 配置文件
- \`start_ai.sh\` - 启动脚本
- \`run_ai.py\` - 主程序
- \`tools.py\` - 工具脚本
- \`switch_model.py\` - 模型切换
- \`fix_gpu.sh\` - GPU修复
- \`venv/\` - Python虚拟环境
- \`model_cache/\` - 模型缓存目录
- \`model_configs/\` - 模型配置目录

## ❓ 常见问题

### 1. 首次运行很慢？
- 首次运行需要下载模型文件
- 下载完成后会缓存，下次启动更快
- 模型大小: ${MODEL_SIZE}

### 2. GPU未检测到？
- 运行: \`python tools.py diagnose\`
- 运行: \`./fix_gpu.sh\`
- 确保已安装NVIDIA驱动和CUDA

### 3. 如何更换模型？
- 运行: \`python switch_model.py\`
- 或启动时指定: \`./start_ai.sh --model "模型名称"\`

### 4. 内存不足？
- 选择更小的模型
- 关闭其他应用程序
- 确保有足够的虚拟内存

## 📄 许可证

本项目基于Apache 2.0许可证开源

## 🤝 支持

如遇问题，请检查:
1. 网络连接是否正常
2. 磁盘空间是否充足
3. Python版本是否为3.8+
4. 查看 \`python tools.py diagnose\` 的输出

EOF

# 创建使用说明简版
cat > QUICKSTART.txt << EOF
快速开始：
1. cd ai_env
2. ./start_ai.sh

常用命令：
- 启动AI: ./start_ai.sh
- 检查环境: python tools.py check
- 诊断GPU: python tools.py diagnose
- 切换模型: python switch_model.py
- 修复GPU: ./fix_gpu.sh

安装信息：
- 目录: ${AI_HOME}
- 模型: ${MODEL_NAME}
- 大小: ${MODEL_SIZE}
- PyTorch: ${PYTORCH_VERSION}
- GPU: ${HAS_NVIDIA}
- CUDA: ${CUDA_VERSION:-未检测到}
- 时间: ${CURRENT_DATE}
EOF

# 添加安装完成信息
echo -e "\n${GREEN}✅ 安装完成！${NC}"
echo "=========================================="
echo "📁 安装目录: ${AI_HOME}"
echo "🤖 选择模型: ${MODEL_NAME}"
echo "📊 模型大小: ${MODEL_SIZE}"
echo "🔧 PyTorch版本: ${PYTORCH_VERSION}"
if [ "$HAS_NVIDIA" = true ]; then
    echo "🖥️  检测到GPU: 是"
    if [ "$HAS_CUDA" = true ] && [ -n "$CUDA_VERSION" ]; then
        echo "⚡ CUDA版本: ${CUDA_VERSION}.x"
    else
        echo "⚠️  CUDA版本: 未检测到或不可用"
    fi
else
    echo "🖥️  检测到GPU: 否"
fi
echo "🚀 启动命令: ./start_ai.sh"
echo "📖 详细说明: 请查看 README.md"
echo "=========================================="
echo -e "\n${BLUE}下一步:${NC}"
echo "1. 进入目录: cd ${AI_HOME}"
echo "2. 启动AI: ./start_ai.sh"
echo "3. 检查环境: python tools.py check"
echo -e "\n${YELLOW}注意: 首次运行需要下载模型（${MODEL_SIZE}），请耐心等待。${NC}"

if [ "$HAS_NVIDIA" = true ] && [ "$PYTORCH_VERSION" = "cpu" ]; then
    echo -e "\n${RED}⚠️  警告: 检测到GPU但选择了CPU版本的PyTorch${NC}"
    echo "💡 建议: 运行 ./fix_gpu.sh 安装GPU版本以获得更好的性能"
fi