"""
GPU工具函数
"""

import torch
import subprocess
import platform
from typing import Dict, List

def check_cuda_version() -> Dict:
    """检查CUDA版本"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': None,
        'cudnn_version': None,
        'gpu_count': 0
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        
        # 尝试获取cuDNN版本
        try:
            import torch.backends.cudnn as cudnn
            info['cudnn_version'] = cudnn.version()
        except:
            pass
    
    return info

def get_system_info() -> Dict:
    """获取系统信息"""
    system = platform.system()
    
    info = {
        'system': system,
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
    }
    
    # Linux特定信息
    if system == 'Linux':
        try:
            dist = platform.freedesktop_os_release()
            info['distribution'] = dist.get('PRETTY_NAME', 'Unknown')
        except:
            pass
    
    return info

def check_gpu_driver() -> Dict:
    """检查GPU驱动"""
    driver_info = {
        'nvidia_driver': None,
        'amd_driver': None,
        'intel_driver': None
    }
    
    system = platform.system()
    
    if system == 'Linux':
        try:
            # 检查NVIDIA驱动
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                driver_info['nvidia_driver'] = result.stdout.strip()
        except:
            pass
    
    return driver_info

def optimize_for_gpu():
    """GPU优化设置"""
    if not torch.cuda.is_available():
        return
    
    # 启用cudnn基准测试（对于固定输入大小）
    torch.backends.cudnn.benchmark = True
    
    # 启用TF32（如果支持）
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("⚡ GPU优化已启用")