"""
模型配置模块
"""

MODEL_CONFIGS = {
    'qwen2.5-0.5b': {
        'name': 'Qwen/Qwen2.5-0.5B-Instruct',
        'type': 'causal_lm',
        'trust_remote_code': True,
        'token': False,
        'max_length': 2048,
        'temperature': 0.7,
        'top_p': 0.95,
        'repetition_penalty': 1.1
    },
    'qwen1.5-1.8b': {
        'name': 'Qwen/Qwen1.5-1.8B-Chat',
        'type': 'causal_lm',
        'trust_remote_code': True,
        'token': False,
        'max_length': 2048
    },
    'phi-2': {
        'name': 'microsoft/phi-2',
        'type': 'causal_lm',
        'trust_remote_code': False,
        'token': False,
        'max_length': 2048
    },
    'tinyllama': {
        'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'type': 'causal_lm',
        'trust_remote_code': False,
        'token': False,
        'max_length': 2048
    }
}

def get_model_config(model_key: str = 'qwen2.5-0.5b') -> Dict:
    """获取模型配置"""
    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS['qwen2.5-0.5b'])
    return config.copy()

def list_available_models() -> List[str]:
    """列出所有可用模型"""
    return list(MODEL_CONFIGS.keys())

def print_model_info(model_key: str):
    """打印模型信息"""
    config = get_model_config(model_key)
    print(f"🤖 模型: {model_key}")
    print(f"  名称: {config['name']}")
    print(f"  类型: {config['type']}")
    print(f"  最大长度: {config.get('max_length', 2048)}")
    return config