#!/usr/bin/env bash
# Launch the Jarvis core (LLM + memory) inside WSL.
set -e
cd "$(dirname "$0")"
exec ./.venv/bin/python server.py
