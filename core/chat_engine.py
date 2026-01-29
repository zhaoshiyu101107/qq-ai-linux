"""
对话引擎模块
"""

import torch
from typing import List, Dict, Any
from core.model_loader import ModelLoader

class ChatEngine:
    """对话引擎"""
    
    def __init__(self, model_key: str = 'qwen2.5-0.5b'):
        self.model_key = model_key
        self.loader = ModelLoader(model_key)
        self.tokenizer = None
        self.model = None
        self.chat_history = []
        
    def initialize(self):
        """初始化对话引擎"""
        print("🚀 初始化对话引擎...")
        self.tokenizer, self.model = self.loader.load()
        print("✅ 对话引擎就绪")
        
    def chat(self, 
             user_input: str, 
             max_tokens: int = 200,
             temperature: float = 0.7,
             top_p: float = 0.95,
             add_to_history: bool = True) -> str:
        """单次对话"""
        if self.model is None:
            self.initialize()
        
        # 准备消息
        messages = [{"role": "user", "content": user_input}]
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码回复
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 清理回复
        response = self._clean_response(response)
        
        # 添加到历史
        if add_to_history:
            self.chat_history.append({
                'user': user_input,
                'assistant': response,
                'tokens': outputs.shape[1]
            })
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """清理回复文本"""
        # 移除提示部分
        markers = ['assistant', 'Assistant:', 'AI:', 'Bot:']
        
        for marker in markers:
            if marker in response:
                response = response.split(marker)[-1].strip()
                break
        
        # 移除可能的特殊标记
        response = response.replace('<|endoftext|>', '').strip()
        
        return response
    
    def interactive_chat(self, 
                        max_tokens: int = 200,
                        temperature: float = 0.7,
                        top_p: float = 0.95):
        """交互式对话"""
        if self.model is None:
            self.initialize()
        
        print("\n" + "="*60)
        print(f"🤖 {self.model_key} 对话模式")
        print("="*60)
        print("💡 命令:")
        print("  /clear  - 清空历史")
        print("  /history - 查看历史")
        print("  /quit   - 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                # 处理命令
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("👋 再见！")
                    break
                elif user_input.lower() == '/clear':
                    self.chat_history.clear()
                    print("🗑️  历史已清空")
                    continue
                elif user_input.lower() == '/history':
                    self.print_history()
                    continue
                elif not user_input:
                    continue
                
                # 生成回复
                print("思考中...", end="", flush=True)
                response = self.chat(
                    user_input, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                print("\r" + " " * 20 + "\r", end="")  # 清除"思考中..."
                print(f"AI: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🔚 退出对话")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
    
    def print_history(self):
        """打印对话历史"""
        if not self.chat_history:
            print("📭 对话历史为空")
            return
        
        print(f"\n📜 对话历史 ({len(self.chat_history)} 条):")
        print("-"*60)
        
        for i, entry in enumerate(self.chat_history, 1):
            print(f"{i}. 你: {entry['user'][:50]}...")
            print(f"   AI: {entry['assistant'][:50]}...")
            print()
    
    def save_history(self, filename: str = "chat_history.txt"):
        """保存对话历史"""
        import json
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
        
        print(f"💾 历史已保存到: {filename}")