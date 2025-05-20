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
        """generate text response"""
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
                        raise Exception(f"Ollama API error: {error_text}")
            except Exception as e:
                print(f"call ollama failed: {e}")
                return ""
                
    async def list_models(self) -> list:
        """get available model list"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.host}/api/tags") as response:
                    if response.status == 200:
                        result = await response.json()
                        return [model['name'] for model in result.get('models', [])]
                    return []
            except Exception as e:
                print(f"get model list failed: {e}")
                return []
                
    async def pull_model(self, model_name: str) -> bool:
        """pull new model"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.host}/api/pull",
                    json={"name": model_name}
                ) as response:
                    return response.status == 200
            except Exception as e:
                print(f"pull model failed: {e}")
                return False
                
    def get_model_info(self) -> Dict[str, Any]:
        """get current model info"""
        return {
            "name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        } 