import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
import httpx
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings

from multimodal_moderation.types.model_choice import ModelChoice


load_dotenv()


def _get_required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"{key} environment variable is required but not set")
    return value


MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "google").strip().lower()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GOOGLE_GEMINI_BASE_URL: str = os.getenv("GOOGLE_GEMINI_BASE_URL", "").strip()
USER_API_KEY: str = _get_required_env("USER_API_KEY")
DEFAULT_GOOGLE_MODEL: str = os.getenv("DEFAULT_GOOGLE_MODEL", "gemini-2.5-flash-lite")

CN_PROVIDER_DEFAULTS = {
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen-plus"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen-plus"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "moonshot-v1-32k"),
    "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "moonshot-v1-32k"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "glm-4-flash"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "glm-4-flash"),
    "volcengine": ("https://ark.cn-beijing.volces.com/api/v3", "VOLCENGINE_API_KEY", "doubao-seed-1-6"),
    "doubao": ("https://ark.cn-beijing.volces.com/api/v3", "VOLCENGINE_API_KEY", "doubao-seed-1-6"),
    "minimax": ("https://api.minimax.chat/v1", "MINIMAX_API_KEY", "MiniMax-M2.7"),
    "stepfun": ("https://api.stepfun.com/v1", "STEPFUN_API_KEY", "step-3.5-flash"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
}

DEFAULT_MODEL: str = os.getenv(
    "DEFAULT_MODEL",
    DEFAULT_GOOGLE_MODEL if MODEL_PROVIDER in {"google", "gemini"} else CN_PROVIDER_DEFAULTS.get(
        MODEL_PROVIDER, ("", "", "qwen-plus")
    )[2],
)
EVAL_JUDGE_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", DEFAULT_MODEL)
EVAL_NUM_REPEATS: int = int(os.getenv("EVAL_NUM_REPEATS", "1"))
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
PHOENIX_URL: str = os.getenv("PHOENIX_URL", "http://127.0.0.1:6006")


def get_model_choice(model_name: str | None = None, provider_name: str | None = None) -> ModelChoice:
    provider_name = (provider_name or MODEL_PROVIDER).strip().lower()
    model_name = model_name or DEFAULT_MODEL

    if provider_name in {"google", "gemini"}:
        gemini_key = _get_required_env("GEMINI_API_KEY") if not GEMINI_API_KEY else GEMINI_API_KEY
        provider_kwargs = {"api_key": gemini_key}
        if GOOGLE_GEMINI_BASE_URL:
            provider_kwargs = {
                "client": genai.Client(
                    api_key=gemini_key,
                    http_options=genai_types.HttpOptions(baseUrl=GOOGLE_GEMINI_BASE_URL),
                )
            }

        return ModelChoice(
            model=GoogleModel(
                model_name or DEFAULT_GOOGLE_MODEL,
                provider=GoogleProvider(**provider_kwargs)
            ),
            model_settings=GoogleModelSettings(google_thinking_config={"thinking_budget": 0}),
        )

    default_base_url, default_key_env, default_model = CN_PROVIDER_DEFAULTS.get(
        provider_name,
        ("", "OPENAI_COMPATIBLE_API_KEY", "qwen-plus"),
    )
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL") or default_base_url
    key_env = os.getenv("OPENAI_COMPATIBLE_API_KEY_ENV") or default_key_env
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.getenv(key_env)
    model_name = model_name or os.getenv("OPENAI_COMPATIBLE_MODEL") or default_model

    if not base_url:
        raise ValueError("OPENAI_COMPATIBLE_BASE_URL must be set for MODEL_PROVIDER=openai-compatible")
    if not api_key:
        raise ValueError(f"{key_env} or OPENAI_COMPATIBLE_API_KEY must be set for MODEL_PROVIDER={provider_name}")

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from openai import AsyncOpenAI

    return ModelChoice(
        model=OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(
                openai_client=AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    http_client=httpx.AsyncClient(trust_env=False),
                ),
            ),
        ),
        model_settings=ModelSettings(),
    )


def get_default_model_choice() -> ModelChoice:
    return get_model_choice()
