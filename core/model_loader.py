"""
模型加载模块
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Dict, Any
from config.model_config import get_model_config
from config.gpu_config import load_gpu_config

class ModelLoader:
    """模型加载器"""
    
    def __init__(self, model_key: str = 'qwen2.5-0.5b'):
        self.model_key = model_key
        self.model_config = get_model_config(model_key)
        self.gpu_config = load_gpu_config()
        self.tokenizer = None
        self.model = None
        
    def load(self) -> Tuple[Any, Any]:
        """加载模型和tokenizer"""
        print(f"🤖 加载模型: {self.model_config['name']}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['name'],
            trust_remote_code=self.model_config['trust_remote_code'],
            token=self.model_config['token']
        )
        
        # 确定数据类型
        torch_dtype = self._get_torch_dtype()
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['name'],
            trust_remote_code=self.model_config['trust_remote_code'],
            token=self.model_config['token'],
            torch_dtype=torch_dtype,
            device_map=self.gpu_config['device_map']
        )
        
        print(f"✅ 模型加载完成")
        print(f"   设备: {self.model.device}")
        print(f"   数据类型: {self.model.dtype}")
        
        return self.tokenizer, self.model
    
    def _get_torch_dtype(self):
        """获取PyTorch数据类型"""
        dtype_str = self.gpu_config.get('torch_dtype', 'float32')
        
        if dtype_str == 'float16':
            return torch.float16
        elif dtype_str == 'bfloat16':
            return torch.bfloat16
        else:
            return torch.float32
    
    def get_memory_usage(self) -> float:
        """获取模型内存使用量（GB）"""
        if self.model is None:
            return 0.0
        
        param_count = sum(p.numel() for p in self.model.parameters())
        
        # 根据数据类型计算内存
        if self.model.dtype == torch.float16:
            bytes_per_param = 2
        elif self.model.dtype == torch.bfloat16:
            bytes_per_param = 2
        else:
            bytes_per_param = 4
        
        memory_gb = (param_count * bytes_per_param) / (1024**3)
        return memory_gb
    
    def unload(self):
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("🗑️  模型已卸载")