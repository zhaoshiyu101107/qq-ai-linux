#!/bin/bash
# 自包含AI环境安装脚本 - 所有内容安装到脚本所在目录

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
echo -e "\n${YELLOW}[1/4] 创建Python虚拟环境...${NC}"
python3 -m venv venv 2>/dev/null || python -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
echo -e "\n${YELLOW}[2/4] 安装Python依赖...${NC}"
pip install --upgrade pip > /dev/null 2>&1

# 安装PyTorch（CPU版本，稳定且体积小）
echo "安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1

# 安装Transformers和其他AI库
echo "安装Transformers..."
pip install transformers accelerate sentencepiece protobuf einops tiktoken > /dev/null 2>&1

# 创建配置文件
echo -e "\n${YELLOW}[3/4] 创建配置文件...${NC}"

cat > config.json << 'EOF'
{
  "environment": "local",
  "model": "Qwen/Qwen3-0.6B-Instruct",
  "install_date": "$(date)",
  "install_dir": "${AI_HOME}",
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
echo -e "\n${YELLOW}[4/4] 创建启动脚本...${NC}"

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
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 50)
print("🤖 Qwen3-0.6B - 本地AI环境")
print("=" * 50)
print(f"目录: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print("=" * 50)
print()

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
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
        if torch.cuda.is_available():
            print("  使用GPU加速")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
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
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                
                print("思考中...", end="", flush=True)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
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

def check_environment():
    """检查环境状态"""
    print("🔍 环境检查")
    print("-" * 40)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 检查其他包
    packages = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("einops", "einops"),
    ]
    
    for name, module in packages:
        try:
            __import__(module)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
    
    print("-" * 40)

def clear_cache():
    """清理模型缓存"""
    import shutil
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    
    if os.path.exists(cache_dir):
        size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                   for f in os.listdir(cache_dir) 
                   if os.path.isfile(os.path.join(cache_dir, f))) / (1024**3)
        
        print(f"缓存大小: {size:.2f} GB")
        response = input("确认删除缓存？(y/N): ").strip().lower()
        
        if response == 'y':
            shutil.rmtree(cache_dir)
            print("✅ 缓存已清理")
        else:
            print("❌ 取消操作")
    else:
        print("✅ 缓存目录不存在")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_environment()
        elif sys.argv[1] == "clear":
            clear_cache()
        else:
            print("用法: python tools.py [check|clear]")
    else:
        check_environment()
EOF

# 创建README文件
cat > README.md << 'EOF'
# 🤖 本地AI环境

这是一个完全自包含的AI运行环境，包含Qwen3-0.6B模型。

## 目录结构