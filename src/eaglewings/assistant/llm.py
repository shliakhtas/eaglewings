"""Яка модель за ключем, який ввела людина. Ключ ніде не зберігається: живе лише в одному запиті.

    sk-ant-…   Anthropic, модель ASSISTANT_ANTHROPIC_MODEL (за замовчуванням Claude Haiku 5.5)
    sk-…       OpenAI, модель ASSISTANT_OPENAI_MODEL (за замовчуванням gpt-4.1-mini)

Усе йде через LiteLLM, тож провайдер це лише назва моделі й ключ.
"""
import os

OPENAI_DEFAULT = "gpt-4.1-mini"
ANTHROPIC_DEFAULT = "anthropic/claude-haiku-5-5"


def provider(key):
    """Провайдер за виглядом ключа; None, якщо ключ не схожий ні на який."""
    key = (key or "").strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    return None


def model(key):
    p = provider(key)
    if p == "anthropic":
        return os.environ.get("ASSISTANT_ANTHROPIC_MODEL", ANTHROPIC_DEFAULT)
    if p == "openai":
        return os.environ.get("ASSISTANT_OPENAI_MODEL", OPENAI_DEFAULT)
    return None


def make(key):
    """LiteLLM для PandasAI з ключем людини; температура 0, щоб однакове питання давало однаковий код."""
    from pandasai_litellm.litellm import LiteLLM
    m = model(key)
    if not m:
        raise ValueError("Ключ не схожий ні на OpenAI (sk-…), ні на Anthropic (sk-ant-…).")
    return LiteLLM(model=m, api_key=key.strip(), temperature=0)
