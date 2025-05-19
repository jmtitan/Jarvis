import aiohttp
import json
from typing import Optional, Dict, Any

class OllamaClient:
    def __init__(self, config: dict):
        self.config = config
        self.host = config['ollama']['host']
        self.model = config['ollama']['model']
        self.temperature = config['ollama']['temperature']
        self.max_tokens = config['ollama']['max_tokens']
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本响应"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            try:
                async with session.post(f"{self.host}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API 错误: {error_text}")
            except Exception as e:
                print(f"调用 Ollama 失败: {e}")
                return ""
                
    async def list_models(self) -> list:
        """获取可用模型列表"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.host}/api/tags") as response:
                    if response.status == 200:
                        result = await response.json()
                        return [model['name'] for model in result.get('models', [])]
                    return []
            except Exception as e:
                print(f"获取模型列表失败: {e}")
                return []
                
    async def pull_model(self, model_name: str) -> bool:
        """拉取新模型"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.host}/api/pull",
                    json={"name": model_name}
                ) as response:
                    return response.status == 200
            except Exception as e:
                print(f"拉取模型失败: {e}")
                return False
                
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        return {
            "name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        } 