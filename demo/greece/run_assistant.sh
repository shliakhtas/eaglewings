#!/usr/bin/env bash
# Асистент застосунку: http://127.0.0.1:8770. Середовище з requirements/assistant.txt бібліотеки й пакетом eaglewings (pip install -e <бібліотека> --no-deps).
cd "$(dirname "$0")" && EAGLEWINGS_PROJECT="$PWD" exec "${ASSISTANT_PYTHON:-python3}" -m uvicorn eaglewings.assistant.server:app --host 127.0.0.1 --port "${ASSISTANT_PORT:-8770}"
