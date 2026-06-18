"""
OpenAI-compatible LLM client (used to talk to a vLLM server).

vLLM exposes an OpenAI-compatible API at e.g. http://127.0.0.1:8000/v1.
This client provides:
  - generate()        : non-streaming, returns the full reply (used by the
                        memory subsystem for summaries / fact extraction)
  - generate_stream() : async generator yielding text deltas as they arrive
                        (used by ChatService for sentence-by-sentence TTS)

Qwen3 ships a hybrid "thinking" mode. For a voice assistant that is pure
latency overhead, so we disable it per-request via
chat_template_kwargs={"enable_thinking": False}.
"""

import json
from typing import Optional, Dict, Any

import aiohttp


class OpenAIClient:
    def __init__(self, config: dict):
        self.config = config
        llm_cfg = config.get("llm", {})
        self.base_url = llm_cfg.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = llm_cfg.get("model", "Qwen3-8B")
        self.temperature = llm_cfg.get("temperature", 0.7)
        self.max_tokens = llm_cfg.get("max_tokens", 1000)
        self.api_key = llm_cfg.get("api_key", "EMPTY")
        self.enable_thinking = bool(llm_cfg.get("enable_thinking", False))
        self.request_timeout = llm_cfg.get("request_timeout", 120)

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_messages(self, prompt: str, system_prompt: Optional[str]):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _payload(self, prompt: str, system_prompt: Optional[str], stream: bool):
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            # Disable Qwen3 chain-of-thought for low-latency voice replies.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        return payload

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Non-streaming generation. Returns the full reply text."""
        url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, json=self._payload(prompt, system_prompt, stream=False),
                    headers=self._headers(),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"vLLM API error ({resp.status}) for {url}: {text}")
                        return ""
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"] or ""
        except aiohttp.ClientConnectorError as e:
            print(f"vLLM connection failed: {e} (is the vLLM server up at {self.base_url}?)")
            return ""
        except Exception as e:
            print(f"vLLM generate failed: {e}")
            return ""

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        """Streaming generation. Yields text deltas (str) as they arrive."""
        url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, json=self._payload(prompt, system_prompt, stream=True),
                    headers=self._headers(),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"vLLM stream API error ({resp.status}) for {url}: {text}")
                        return
                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except Exception:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        piece = delta.get("content")
                        if piece:
                            yield piece
        except aiohttp.ClientConnectorError as e:
            print(f"vLLM connection failed: {e} (is the vLLM server up at {self.base_url}?)")
            return
        except Exception as e:
            print(f"vLLM stream failed: {e}")
            return

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "backend": "vllm",
        }

    async def close(self):
        # Sessions are created per-request, nothing persistent to clean up.
        print("OpenAIClient closed.")
