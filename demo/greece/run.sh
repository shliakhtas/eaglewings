#!/usr/bin/env bash
# Сайт проєкту: http://127.0.0.1:8766 (лендінг, застосунок, дослідження; /ua/ і /en/). Перебудувати: eaglewings run site_data
cd "$(dirname "$0")" && exec "${PYTHON:-python3}" -m eaglewings.cli serve --port "${PORT:-8766}"
