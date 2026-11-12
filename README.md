# EagleWings

**A ready-made data-analytics platform for a science manager.** AI agents do the runs; a person asks the questions and owns the conclusion. EagleWings gives you the data pipeline, the checks, the pages and a personal AI assistant, for media, analytics startups and research teams.

**Live:** https://eaglewings.noosvit.com · demo case: [product landing](https://eaglewings.noosvit.com/landing.html) · [app with AI assistant](https://eaglewings.noosvit.com/app.html) · [research page](https://eaglewings.noosvit.com/science/)

> *"As an eagle stirreth up her nest, fluttereth over her young, spreadeth abroad her wings, taketh them, beareth them on her wings"* (Deuteronomy 32:11). The platform carries a young study on ready infrastructure.

## What you get

| Part | What it does |
|---|---|
| **Pipeline** | One [Prefect](https://www.prefect.io/) flow: raw data → versioned datasets → research branches → winner models → pages → docs. Rerun only what changed: `eaglewings run --from <branch>`. |
| **Branches** | Each research question is a branch with several competing algorithms. The leader is chosen on validation only; the test result comes with a 95 % bootstrap interval. Registry: [`src/eaglewings/ds/branches.py`](src/eaglewings/ds/branches.py). |
| **Datasets and runs** | Every dataset is versioned by content and parameters (`data/datasets/<name>/vN/manifest.json`); every run writes a run version with numbers and a diff against the previous one (`data/runs/`). Optional tracking in [ClearML](https://clear.ml). |
| **Research page** | Questions, data, approaches, results and intervals for every branch, built automatically from the same pipeline. |
| **App** | Map, analytics, monitoring stream and an AI assistant that answers questions about your data in plain language and draws maps, tables and charts (PandasAI in a sandbox; bring your own OpenAI or Anthropic key). |
| **Landing** | A product landing assembled from blocks, with recordings of the real app. |
| **Two languages** | Every page in Ukrainian and English (`/ua/`, `/en/`). |
| **Claude Code skills** | [`.claude/skills/`](.claude/skills): `new-study` (start a study from the base branches), `extend-study` (new algorithm or branch), `retarget-study` (new period or territory), `study-docs` (paper, limitations, sources, README). |

## Demo case: Forests of Thrace

[`demo/greece/`](demo/greece) is a complete project built with EagleWings: three national parks of Thrace, Greece (Dadia–Lefkimi–Soufli, Evros Delta, Eastern Macedonia and Thrace), 2021–2025, open satellite data only: FIRMS, Hansen GFC, EFFIS, Sentinel-2, NASA OPERA, NASA POWER, SRTM, OpenStreetMap. 16 research questions on fires, logging, detectors and monitoring, each answered per park with intervals, and negative results kept.

Results, claims and their limits: [demo/greece/README.md](demo/greece/README.md) and [demo/greece/docs/LIMITATIONS.md](demo/greece/docs/LIMITATIONS.md) (case documents are in Ukrainian).

## Repository layout

| Path | What is there |
|---|---|
| [`src/eaglewings/`](src/eaglewings) | the library: `ds/` (branch registry, tasks, metrics, datasets, tracking), `scripts/` (pipeline steps), `flows/pipeline.py` (Prefect flow), `assistant/` (chat server and sandbox), `templates/` (app, research page, landing), `cli.py` |
| [`demo/greece/`](demo/greece) | the demo project: `config/`, `docs/`, `models/` (winner cards), `notebooks/`, landing content in `web/landing/`; `data/` and built pages are created by `eaglewings run` |
| [`requirements/`](requirements) | pinned environments: `ds.txt` (pipeline), `cv.txt` (detectors, GPU), `assistant.txt` (chat) |
| [`docs/architecture.html`](docs/architecture.html) | architecture diagram |
| [`.claude/skills/`](.claude/skills) | Claude Code skills for new and changed studies |
| `.env.example` | optional settings |

## Install

Python 3.11, Linux or macOS. Two environments, because the assistant (PandasAI 3) needs pandas 2.x.

```bash
git clone https://github.com/shliakhtas/eaglewings.git && cd eaglewings

# pipeline, site, CLI
python3.11 -m venv .venv
.venv/bin/pip install -r requirements/ds.txt
.venv/bin/pip install -e . --no-deps

# optional: detectors and vision demo (GPU with CUDA 12.8; YOLO is AGPL-3.0, see NOTICE)
.venv/bin/pip install -r requirements/cv.txt

# optional: AI assistant for the app chat
python3.11 -m venv .venv-api
.venv-api/bin/pip install -r requirements/assistant.txt
.venv-api/bin/pip install -e . --no-deps
```

## Quick start

To just look at the demo, open the live site: https://eaglewings.noosvit.com. The repository holds the code, config and documents of the case, not its data or built pages (about 220 MB): the pipeline downloads open data and rebuilds everything.

Build the demo locally (hours; detectors need a GPU, everything else runs on CPU):

```bash
source .venv/bin/activate && cd demo/greece
prefect server start --host 127.0.0.1 --port 4200   # separate terminal; flow UI at http://127.0.0.1:4200
eaglewings run --plan            # every step and dependency, nothing runs
eaglewings run                   # data → datasets → branches → models → site → docs, then a run version
eaglewings run FG_risk LD        # only these branches
eaglewings run --from LD         # LD changed: rerun it and everything downstream
eaglewings run core --clearml    # same, also logged to ClearML
eaglewings serve                 # site: http://127.0.0.1:8766  (landing.html, app.html, science/, /en/)
```

Chat in the app needs the assistant server, from the assistant environment: `../../.venv-api/bin/eaglewings assistant` (http://127.0.0.1:8770), then paste your OpenAI or Anthropic key in the app.

Step by step: [demo/greece/docs/REPRODUCE.md](demo/greece/docs/REPRODUCE.md).

## Configuration

Nothing secret is needed to run EagleWings:

- **Data**: all sources are open and downloaded without keys.
- **Chat**: the viewer types an OpenAI (`sk-…`) or Anthropic (`sk-ant-…`) key in the app. It lives only in that browser tab and in the request; the server never stores or logs it.
- **ClearML** (optional): only with `--clearml`. Keys go to `~/clearml.conf` (`clearml-init`) or to `CLEARML_API_ACCESS_KEY` / `CLEARML_API_SECRET_KEY`. Without `--clearml` everything is logged locally in `logs/runs/`, even if keys exist.

Optional settings (assistant models, CORS origins, sandbox isolation, park subset) are listed in [`.env.example`](.env.example). The code does not read `.env` by itself; load it into the shell:

```bash
cp .env.example .env            # edit, never commit (.env is git-ignored)
set -a && . ./.env && set +a
```

## Commands

| Command | What it does |
|---|---|
| `eaglewings init <dir> [--from demo/greece]` | new project: config, landing content, translation, LIMITATIONS; no data |
| `eaglewings plan` | pipeline steps in plain words, nothing runs |
| `eaglewings run [steps\|branches] [--from X] [--clearml]` | the Prefect flow |
| `eaglewings script <name> [args]` | one library script (`scripts/<name>.py`) in the project folder, as the flow runs it |
| `eaglewings serve [--port 8766]` | the project site from `web/` |
| `eaglewings assistant [--port 8770]` | the chat server (assistant environment) |

Run them from a project folder (the one with `eaglewings.toml`) or set `EAGLEWINGS_PROJECT`.

## Start your own study

```bash
eaglewings init ../my-study --from demo/greece
cd ../my-study
```

Then open the folder in [Claude Code](https://claude.com/claude-code) and run the `new-study` skill: it asks which questions, period and territory you need, switches on the branches, sets up the territory, builds the landing block by block, runs the flow and builds the pages.

## Data and licenses

- **Code**: [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
- **Attribution**: derivative works and redistributions are asked to show **"Built with EagleWings"**.
- **Third-party packages** keep their own licenses and are not bundled. The optional vision branches use Ultralytics YOLO (AGPL-3.0); the core pipeline does not need it.
- **Data of the demo** is not stored in the repository; the pipeline downloads it from open sources under their own terms: NASA (FIRMS, MODIS, POWER, OPERA), Hansen GFC (CC BY 4.0), Copernicus (Sentinel-2, EFFIS), OpenStreetMap (ODbL), Wikimedia Commons photos (CC BY-SA, authors shown on the site). Benchmarks (D-Fire, VEDAI, ENA24) and third-party detector weights are not redistributed: the pipeline downloads them from the original sources. Full list with DOIs: [demo/greece/docs/SOURCES.md](demo/greece/docs/SOURCES.md).

## Citation

If you use EagleWings in research, cite it as in [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Use of AI

Code, analysis and texts were prepared with the help of the AI assistant Claude (Anthropic). Protocols, checks of the numbers and conclusions were set and reviewed by a person.
