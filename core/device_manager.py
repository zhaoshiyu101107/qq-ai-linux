"""
设备管理模块
"""

import torch
from typing import Dict, List, Any
from config.gpu_config import detect_gpus, get_gpu_memory_info

class DeviceManager:
    """设备管理器"""
    
    def __init__(self):
        self.gpus = detect_gpus()
        self.config = None
        
    def print_device_info(self):
        """打印设备信息"""
        print("=" * 60)
        print("🎮 设备检测报告")
        print("=" * 60)
        
        if not self.gpus:
            print("❌ 未检测到GPU设备")
            print("💡 将使用CPU运行")
            return False
        
        print(f"✅ 检测到 {len(self.gpus)} 个GPU:")
        print("-" * 60)
        
        for gpu in self.gpus:
            mem_info = get_gpu_memory_info(gpu['id'])
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"     显存: {mem_info.get('allocated_gb', 0):.1f}/{gpu['memory_total_gb']:.1f} GB")
            print(f"     算力: CUDA {gpu['capability']}")
            print()
        
        return True
    
    def get_user_choice(self) -> Dict:
        """获取用户设备选择"""
        if not self.gpus:
            return {
                'device': 'cpu',
                'device_map': 'cpu',
                'torch_dtype': torch.float32,
                'use_gpu': False
            }
        
        print("\n选择GPU使用方式:")
        print("1. 🚀 自动选择最佳GPU")
        print("2. 🔢 手动选择GPU")
        print("3. 💻 强制使用CPU")
        print("-" * 40)
        
        while True:
            choice = input("请选择 (1-3, 默认: 1): ").strip()
            
            if not choice:
                choice = "1"
            
            if choice == "1":
                return self._auto_select()
            elif choice == "2":
                return self._manual_select()
            elif choice == "3":
                return {
                    'device': 'cpu',
                    'device_map': 'cpu',
                    'torch_dtype': torch.float32,
                    'use_gpu': False
                }
            else:
                print("❌ 无效选择，请重新输入")
    
    def _auto_select(self) -> Dict:
        """自动选择GPU"""
        best_gpu = max(self.gpus, key=lambda x: x['memory_total_gb'])
        
        print(f"\n✅ 自动选择: GPU {best_gpu['id']} ({best_gpu['name']})")
        print(f"   显存: {best_gpu['memory_total_gb']:.1f} GB")
        
        return {
            'device': f"cuda:{best_gpu['id']}",
            'device_map': f"cuda:{best_gpu['id']}",
            'torch_dtype': torch.float16,
            'use_gpu': True,
            'selected_gpu_id': best_gpu['id']
        }
    
    def _manual_select(self) -> Dict:
        """手动选择GPU"""
        print("\n可用GPU:")
        for gpu in self.gpus:
            print(f"  [{gpu['id']}] {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)")
        
        while True:
            try:
                selection = input("\n选择GPU (输入序号, 如: 0): ").strip()
                gpu_id = int(selection)
                
                if gpu_id not in [g['id'] for g in self.gpus]:
                    raise ValueError(f"GPU {gpu_id} 不存在")
                
                selected_gpu = next(g for g in self.gpus if g['id'] == gpu_id)
                print(f"✅ 选择: GPU {gpu_id} ({selected_gpu['name']})")
                
                return {
                    'device': f"cuda:{gpu_id}",
                    'device_map': f"cuda:{gpu_id}",
                    'torch_dtype': torch.float16,
                    'use_gpu': True,
                    'selected_gpu_id': gpu_id
                }
                
            except (ValueError, TypeError) as e:
                print(f"❌ 错误: {e}")
                print("请重新输入")