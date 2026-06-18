"""Smoke test: drive the WSL core from the Windows CoreClient."""
import asyncio
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core_client import CoreClient


async def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    client = CoreClient(config)
    await client.connect()

    reply = await client.chat("Say hello in exactly five words.")
    print("CHAT REPLY:", repr(reply))

    stats = await client.get_stats()
    print("STATS:", stats)

    await client.close()
    print("OK" if reply else "EMPTY REPLY")


asyncio.run(main())
