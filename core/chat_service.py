"""
ChatService: the core "brain" of Jarvis that runs inside WSL.

It owns the LLM client (vLLM, OpenAI-compatible) and the memory subsystem.

Two entry points:
  - handle(text)        : non-streaming, returns the full reply (kept for
                          simple request/response use and testing)
  - handle_stream(text) : async generator yielding *complete sentences* as the
                          LLM streams them, so the desktop side can synthesize
                          and play each sentence while the next is generated.

After a full streamed response, the interaction is stored in memory.
"""

import re
from pathlib import Path

from llm.openai_client import OpenAIClient
from memory.memory_manager import MemoryManager

# Sentence boundary: end-of-sentence punctuation (EN + CN) optionally followed
# by closing quotes/brackets. We also flush on newlines.
_SENTENCE_END = re.compile(r'([.!?。！？…]+["”\')\]]?\s|[\n]+)')


class ChatService:
    def __init__(self, config: dict):
        self.config = config
        self.memory_enabled = bool(config.get("memory", {}).get("enabled", False))

        self.llm = OpenAIClient(config)
        self.memory = MemoryManager(config) if self.memory_enabled else None
        if self.memory:
            self.memory.set_llm_client(self.llm)

        # Load the system prompt once (same file the Windows side used)
        self.system_prompt = None
        prompt_path = Path("prompts/Emma.md")
        if prompt_path.is_file():
            try:
                self.system_prompt = prompt_path.read_text(encoding="utf-8")
                print(f"Loaded system prompt from {prompt_path}")
            except Exception as e:
                print(f"Warning: could not read system prompt {prompt_path}: {e}")

    async def _build_prompt(self, text: str) -> str:
        if self.memory:
            try:
                prompt = await self.memory.build_enhanced_prompt(text)
                print(f"Using memory-enhanced prompt (length: {len(prompt)} chars)")
                return prompt
            except Exception as e:
                print(f"Memory prompt build failed, falling back to raw text: {e}")
        return text

    async def _store(self, text: str, reply: str):
        if self.memory and reply:
            try:
                await self.memory.process_interaction(text, reply)
                print("Interaction stored in memory")
            except Exception as e:
                print(f"Error storing interaction in memory: {e}")

    async def handle(self, text: str) -> str:
        """Non-streaming: build prompt, generate, store, return full reply."""
        text = (text or "").strip()
        if not text:
            return ""
        prompt = await self._build_prompt(text)
        reply = await self.llm.generate(prompt, system_prompt=self.system_prompt)
        if not reply:
            print("LLM returned empty response")
            return ""
        await self._store(text, reply)
        return reply

    async def handle_stream(self, text: str):
        """Streaming: yield complete sentences as they are produced.

        Strips any Qwen3 <think>...</think> spans defensively, segments the
        token stream into sentences, and stores the full reply in memory once
        the stream finishes.
        """
        text = (text or "").strip()
        if not text:
            return
        prompt = await self._build_prompt(text)

        buffer = ""          # not-yet-flushed sentence text
        full_parts = []      # complete reply (for memory)
        in_think = False     # inside a <think> span

        async for piece in self.llm.generate_stream(prompt, system_prompt=self.system_prompt):
            # Defensive <think> stripping (should be off via enable_thinking=False)
            if "<think>" in piece or "</think>" in piece or in_think:
                piece, in_think = self._strip_think(piece, in_think)
                if not piece:
                    continue

            buffer += piece
            # Emit every complete sentence currently in the buffer
            while True:
                m = _SENTENCE_END.search(buffer)
                if not m:
                    break
                end = m.end()
                sentence = buffer[:end].strip()
                buffer = buffer[end:]
                if sentence:
                    full_parts.append(sentence)
                    yield sentence

        # Flush any trailing partial sentence
        tail = buffer.strip()
        if tail:
            full_parts.append(tail)
            yield tail

        full_reply = " ".join(full_parts).strip()
        await self._store(text, full_reply)

    @staticmethod
    def _strip_think(piece: str, in_think: bool):
        """Remove content inside <think>...</think>. Returns (clean, in_think)."""
        out = []
        i = 0
        while i < len(piece):
            if in_think:
                close = piece.find("</think>", i)
                if close == -1:
                    return "".join(out), True
                i = close + len("</think>")
                in_think = False
            else:
                open_ = piece.find("<think>", i)
                if open_ == -1:
                    out.append(piece[i:])
                    break
                out.append(piece[i:open_])
                i = open_ + len("<think>")
                in_think = True
        return "".join(out), in_think

    def stats(self) -> dict:
        info = {"model": self.llm.get_model_info()}
        if self.memory:
            info["memory"] = self.memory.get_memory_stats()
        return info

    async def shutdown(self):
        try:
            if self.memory:
                await self.memory.shutdown()
        finally:
            await self.llm.close()
