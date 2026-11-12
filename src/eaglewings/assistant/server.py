"""Сервер асистента для застосунку НП Дадья: питання звичайною мовою → таблиця й об'єкти на мапі.

    eaglewings assistant                    # або ./run_assistant.sh у теці проєкту

    GET  /health                       чи сервер живий
    POST /check {key}                  чи ключ робочий (запит в один токен): невалідний, без коштів, ок
    POST /ask  {question, key, recent, park} відповідь про вибраний парк у форматі чату застосунку: text, type, table, rows, geojson, chart

Ключ OpenAI чи Anthropic приходить у тілі запиту, використовується лише для цього запиту й ніде не пишеться:
ні на диск, ні в журнал. Без ключа застосунок працює як раніше, з готовими питаннями, і цей сервер йому не потрібен.
"""
import logging
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import agent

os.makedirs(agent.EXPORTS, exist_ok=True)
os.chdir(agent.EXPORTS)                     # PandasAI пише графіки у відносний exports/charts: тримаємо їх у logs/assistant
logging.getLogger("uvicorn.access").disabled = True     # тіло запиту з ключем і так не журналюється; шлях теж не потрібен
LOCK = threading.Lock()                     # одне питання за раз: форк пісочниці з потоками безпечніший послідовно

app = FastAPI(title="Асистент EagleWings")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ASSISTANT_ORIGINS", "*").split(","), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class Ask(BaseModel):
    question: str = Field(max_length=1000)
    key: str = Field(max_length=400)
    recent: list = Field(default_factory=list)
    park: str = Field(default="", max_length=40)          # вибраний у застосунку парк; порожньо — основний
    lang: str = Field(default="uk", max_length=4)         # мова відповіді: uk чи en (версія сайту /ua/ чи /en/)


@app.get("/health")
def health():
    return dict(status="ok", parks=list(agent.PARKS), tables={n: len(f) for n, f in agent.FRAMES.items()}, now=agent.NOW)


@app.post("/ask")
def ask(req: Ask):
    with LOCK:
        return agent.ask(req.question, req.key, req.recent, req.park or None, "en" if req.lang == "en" else "uk")


class Key(BaseModel):
    key: str = Field(max_length=400)


@app.post("/check")
def check(req: Key):
    return agent.check(req.key)
