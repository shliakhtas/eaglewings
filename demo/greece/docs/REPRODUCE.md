# Відтворення: Ліси Фракії

Запускати з теки цього проєкту (`demo/greece`). Кожен прогін записує свою версію в `data/runs/` і показує, що змінилось порівняно з попереднім.

## Середовище

| | |
|---|---|
| Python | 3.11 |
| Базові залежності | `requirements/ds.txt` бібліотеки (закріплені версії) |
| Комп'ютерний зір | `requirements/cv.txt`, GPU з CUDA 12.8 |
| Асистент чату | `requirements/assistant.txt`, окреме середовище (PandasAI 3 потребує pandas 2.x) |
| Перевірено | чисте віртуальне середовище Python 3.11.11, `pip check` без конфліктів |

Встановлення: розділ «Встановлення» в README бібліотеки.

```bash
prefect server start --host 127.0.0.1 --port 4200   # в окремому терміналі
eaglewings run --plan        # усі кроки й залежності, нічого не запускаючи
eaglewings run data          # сирі набори даних і відкриті еталони
eaglewings run vision        # комп'ютерний зір на справжніх фото, відео й знімках (GPU)
eaglewings run mocks         # сценарний потік сигналів демо, бере справжній дим із vision
eaglewings run datasets      # датасети з версіями
eaglewings run core          # гілки без GPU
eaglewings run cv            # SD, CD і демо детекторів, GPU, ~2 год
eaglewings run models        # моделі-переможці
eaglewings run site_data     # дані для сайту: застосунок, сторінка досліджень, EDA, мапи
eaglewings run notebooks     # ноутбуки
eaglewings run               # усе по черзі, наприкінці версія прогону
eaglewings serve             # сайт, http://127.0.0.1:8766
```

Трекінг локальний, у `logs/runs/`. У ClearML лише явно: `eaglewings run <гілка> --clearml`, ключі в `~/clearml.conf`.

## Звідки дані

| Набір | Джерело | Ліцензія або умови |
|---|---|---|
| FIRMS, MODIS, POWER | NASA | відкриті |
| Hansen GFC v1.12 | UMD / Google / USGS / NASA | CC BY 4.0 |
| OSM | OpenStreetMap | ODbL |
| EFFIS Burnt Areas | Copernicus EMS | відкриті з посиланням на джерело |
| D-Fire | Venâncio et al. 2022, дзеркало Hugging Face `badsaarow/d-fire` | див. репозиторій авторів |
| VEDAI | Razakarivony & Jurie 2016, дзеркало `ckyrkou/vedai` | CC BY 4.0 (за даними дзеркала) |
| ENA24 | Yousif et al. 2019, LILA BC, дзеркало `mbiarreta/camera-trap-lila-ena24` | див. LILA BC |
