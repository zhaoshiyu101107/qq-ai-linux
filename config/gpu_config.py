"""
GPU配置模块 - 检测和配置GPU设备
"""

import torch
import json
from datetime import datetime
from typing import Dict, List, Union

def detect_gpus() -> List[Dict]:
    """检测所有GPU设备"""
    if not torch.cuda.is_available():
        return []
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'id': i,
            'name': props.name,
            'memory_total_gb': props.total_memory / 1024**3,
            'capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count,
            'is_available': True
        })
    
    return gpus

def get_gpu_memory_info(gpu_id: int) -> Dict:
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return {}
    
    torch.cuda.synchronize(gpu_id)
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'free_gb': reserved - allocated if reserved > allocated else 0
    }

def save_gpu_config(config: Dict, gpus: List[Dict], filename: str = "gpu_config.json"):
    """保存GPU配置到文件"""
    config_data = {
        'config': config,
        'gpu_info': gpus,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
    }
    
    with open(filename, 'w') as f:
        json.dump(config_data, f, indent=2, default=str)
    
    return filename

def load_gpu_config(filename: str = "gpu_config.json") -> Dict:
    """从文件加载GPU配置"""
    try:
        with open(filename, 'r') as f:
            config_data = json.load(f)
        return config_data['config']
    except FileNotFoundError:
        print(f"⚠️  配置文件 {filename} 未找到，使用默认CPU配置")
        return get_default_config()
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return get_default_config()

def get_default_config() -> Dict:
    """获取默认CPU配置"""
    return {
        'device': 'cpu',
        'device_map': 'cpu',
        'torch_dtype': 'float32',
        'use_gpu': False
    }

def select_best_gpu(gpus: List[Dict]) -> Dict:
    """选择性能最好的GPU"""
    if not gpus:
        return get_default_config()
    
    # 按显存大小选择
    best_gpu = max(gpus, key=lambda x: x['memory_total_gb'])
    
    return {
        'device': f"cuda:{best_gpu['id']}",
        'device_map': f"cuda:{best_gpu['id']}",
        'torch_dtype': 'float16',
        'use_gpu': True,
        'selected_gpu': best_gpu
    }