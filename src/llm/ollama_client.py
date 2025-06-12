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
        
        # Don't create connector in __init__, create fresh ones per session
        self.connector = None
        self._session = None

    def _create_connector(self):
        """Create a new connector"""
        return aiohttp.TCPConnector(ssl=False)

    async def _get_session(self):
        """Get or create a session with a fresh connector"""
        # Always create a new session with a new connector to avoid "Session is closed" errors
        connector = self._create_connector()
        return aiohttp.ClientSession(connector=connector)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """generate text response"""
        # Use `async with await self._get_session() ...` to ensure session is properly closed
        try:
            async with await self._get_session() as session: 
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
                    url = f"{self.host.rstrip('/')}/api/generate"
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get('response', '')
                        else:
                            error_text = await response.text()
                            print(f"Ollama API error ({response.status}) for {url}: {error_text}")
                            raise Exception(f"Ollama API error: {response.status} - {error_text}")
                except aiohttp.ClientConnectorError as e: 
                    print(f"Ollama connection failed (ClientConnectorError): {e}")
                    print(f"Attempted to connect to: {self.host}")
                    return ""
                except Exception as e:
                    print(f"Call to Ollama generate failed: {e}")
                    return ""
        except Exception as e:
            print(f"Session creation failed: {e}")
            return ""
                
    async def list_models(self) -> list:
        """get available model list"""
        try:
            async with await self._get_session() as session: 
                try:
                    url = f"{self.host.rstrip('/')}/api/tags"
                    async with session.get(url) as response:
                        if response.status == 200:
                            result = await response.json()
                            return [model_info['name'] for model_info in result.get('models', [])]
                        else:
                            print(f"Ollama list_models API error ({response.status}): {await response.text()}")
                            return []
                except aiohttp.ClientConnectorError as e:
                    print(f"Ollama connection failed during list_models (ClientConnectorError): {e}")
                    return []
                except Exception as e:
                    print(f"Get model list failed: {e}")
                    return []
        except Exception as e:
            print(f"Session creation failed in list_models: {e}")
            return []
                
    async def pull_model(self, model_name: str) -> bool:
        """pull new model"""
        try:
            async with await self._get_session() as session: 
                try:
                    url = f"{self.host.rstrip('/')}/api/pull"
                    async with session.post(url, json={"name": model_name}) as response:
                        if response.status == 200:
                            print(f"Pulling model {model_name} started (response: {response.status})")
                            # Simplified: assumes 200 means pull will proceed.
                            # Robust handling requires processing the event stream from Ollama.
                            return True 
                        else:
                            print(f"Ollama pull_model API error ({response.status}): {await response.text()}")
                            return False
                except aiohttp.ClientConnectorError as e:
                    print(f"Ollama connection failed during pull_model (ClientConnectorError): {e}")
                    return False
                except Exception as e:
                    print(f"Pull model failed: {e}")
                    return False
        except Exception as e:
            print(f"Session creation failed in pull_model: {e}")
            return False
                
    def get_model_info(self) -> Dict[str, Any]:
        """get current model info"""
        return {
            "name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    async def close(self):
        """Close any persistent resources. Since we create sessions per-request, this is mostly a no-op."""
        # Clean up any persistent state if needed
        self.connector = None
        self._session = None
        print("OllamaClient closed.") 