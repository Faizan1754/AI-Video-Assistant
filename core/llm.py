"""LLM access for the whole project - one place to choose the provider.

Pick the provider in .env:

    LLM_PROVIDER=gemini        # gemini | groq | mistral | ollama
    GEMINI_API_KEY="..."       # GROQ_API_KEY / MISTRAL_API_KEY for the others

If LLM_PROVIDER is not set, the first key found is used (gemini, groq, mistral).
Optional: GEMINI_MODEL / GROQ_MODEL / MISTRAL_MODEL / OLLAMA_MODEL,
GEMINI_FALLBACK_MODELS / GROQ_FALLBACK_MODELS (comma-separated, tried when the main
model keeps failing; set to an empty value to turn fallbacks off),
LLM_MIN_INTERVAL (seconds between requests), LLM_MAX_ATTEMPTS.
"""
import os
import random
import threading
import time
from functools import lru_cache

import httpx

# Free-tier friendly defaults. Model names change often - override with *_MODEL.
_PROVIDERS = {
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "key_env_alt": "GOOGLE_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.1-flash-lite",
        "fallback_models": ["gemini-3.5-flash-lite"],   # each model has its own free quota
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "interval": 6.5,   # free tier is ~10 requests/minute
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",         # replaced Llama 3.3 70B (shut down Aug 2026)
        "fallback_models": ["openai/gpt-oss-20b"],      # smaller, separate limit pool
        "base_url": "https://api.groq.com/openai/v1",
        "interval": 2.1,   # free tier is ~30 requests/minute
    },
    "mistral": {
        "key_env": "MISTRAL_API_KEY",
        "model_env": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
        "interval": 31.0,  # free "Experiment" plan is ~2 requests/minute
    },
    "ollama": {
        "key_env": None,
        "model_env": "OLLAMA_MODEL",
        "default_model": "llama3.2",
        "base_url": "http://localhost:11434/v1",  # override with OLLAMA_BASE_URL
        "interval": 0.0,   # local - no limits
    },
}
_AUTO_ORDER = ("gemini", "groq", "mistral")

_NO_KEY_HELP = (
    "No LLM API key found. Add one to your .env file, for example:\n"
    "  LLM_PROVIDER=gemini\n"
    "  GEMINI_API_KEY=your-key   (free key: https://aistudio.google.com/apikey)\n"
    "Other options: GROQ_API_KEY (console.groq.com), MISTRAL_API_KEY, "
    "or LLM_PROVIDER=ollama for a local model."
)

_last_call = 0.0
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Provider selection (all env reads happen at call time, so .env order is safe)
# --------------------------------------------------------------------------
def _api_key(provider: str):
    cfg = _PROVIDERS[provider]
    for name in (cfg.get("key_env"), cfg.get("key_env_alt")):
        if name:
            value = (os.getenv(name) or "").strip()
            if value:
                return value
    return None


def current_provider() -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit not in _PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{explicit}'. Use one of: {', '.join(_PROVIDERS)}."
            )
        return explicit
    for name in _AUTO_ORDER:
        if _api_key(name):
            return name
    raise RuntimeError(_NO_KEY_HELP)


def _model_name(provider: str) -> str:
    cfg = _PROVIDERS[provider]
    return (os.getenv(cfg["model_env"]) or "").strip() or cfg["default_model"]


def _min_interval(provider: str) -> float:
    override = os.getenv("LLM_MIN_INTERVAL")
    if override is not None and override.strip() != "":
        return float(override)
    return _PROVIDERS[provider]["interval"]


def _max_attempts() -> int:
    return max(1, int(os.getenv("LLM_MAX_ATTEMPTS", "6")))


def _throttle() -> None:
    """Keep at least the provider's minimum gap between request starts, so the
    free-tier requests-per-minute limit is not exceeded in the first place."""
    global _last_call
    interval = _min_interval(current_provider())
    if interval <= 0:
        return
    with _lock:
        wait = _last_call + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _rate_limit_help(provider: str) -> str:
    return (
        f"{provider} is still rate-limiting requests (HTTP 429) after several retries. "
        "Wait a minute and try again, or switch provider with LLM_PROVIDER in .env "
        "(free tiers have strict request/token limits - check your provider's console). "
        "You can also raise LLM_MAX_ATTEMPTS."
    )


def _auth_help(provider: str) -> str:
    cfg = _PROVIDERS[provider]
    return (
        f"{provider} rejected the API key. Check {cfg['key_env']} in your .env "
        "(no extra spaces, the key must be inside the quotes), then restart the app."
    )


# --------------------------------------------------------------------------
# Mistral (its LangChain wrapper has its retry switched off, so add our own)
# --------------------------------------------------------------------------
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Honour the server's Retry-After header, else exponential backoff + jitter."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(max(float(header), 1.0), 60.0)
        except ValueError:
            pass
    return min(2 ** attempt, 30) + random.uniform(0, 1)


@lru_cache(maxsize=1)
def _mistral_class():
    from langchain_mistralai import ChatMistralAI

    class RateLimitedMistral(ChatMistralAI):
        def completion_with_retry(self, *args, **kwargs):
            max_attempts = _max_attempts()
            for attempt in range(1, max_attempts + 1):
                _throttle()
                try:
                    return super().completion_with_retry(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code
                    if status == 401:
                        raise RuntimeError(_auth_help("mistral")) from e
                    if status not in _RETRY_STATUS:
                        raise  # e.g. 400 bad request: retrying won't help
                    if attempt == max_attempts:
                        if status == 429:
                            raise RuntimeError(_rate_limit_help("mistral")) from e
                        raise
                    delay = _retry_delay(e.response, attempt)
                    print(f"mistral returned {status}; retrying in {delay:.0f}s "
                          f"(attempt {attempt}/{max_attempts})...")
                    time.sleep(delay)

    return RateLimitedMistral


# --------------------------------------------------------------------------
# Gemini / Groq / Ollama through their OpenAI-compatible endpoints
# --------------------------------------------------------------------------
class _Exhausted(Exception):
    """All retries for one model were used up (rate limit, daily quota or overload)."""

    def __init__(self, error, daily=False):
        super().__init__(str(error))
        self.error = error
        self.daily = daily


def _fallback_models(provider: str, primary: str) -> list:
    env = os.getenv(f"{provider.upper()}_FALLBACK_MODELS")
    if env is not None:   # set (even blank) -> the user decides
        names = [m.strip() for m in env.split(",") if m.strip()]
    else:
        names = list(_PROVIDERS[provider].get("fallback_models", []))
    return [m for m in dict.fromkeys(names) if m != primary]


def _is_daily_quota(error) -> bool:
    """Daily quotas will not clear within seconds, so do not wait on them."""
    text = str(error).lower()
    return "perday" in text or "per day" in text or "daily" in text


def _backoff(error, attempt: int) -> float:
    """Retry-After if the server sent one, else 4s, 8s, 16s, 32s, 40s... (+ jitter).
    Overload (503) usually needs far longer than a few seconds to clear."""
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return min(max(float(header), 1.0), 60.0)
        except ValueError:
            pass
    import openai

    # connection blips clear faster than provider overload
    base, cap = (2, 15) if isinstance(error, openai.APIConnectionError) else (4, 40)
    return min(base * 2 ** (attempt - 1), cap) + random.uniform(0, 1)


def _cause_text(error) -> str:
    cause = getattr(error, "__cause__", None)
    return f"{type(cause).__name__}: {cause}"[:200] if cause is not None else ""


def _available_models(client) -> list:
    """Best-effort list of chat model ids the provider offers right now."""
    skip = ("whisper", "guard", "tts", "embed", "orpheus", "playai", "moderation", "transcribe", "image")
    try:
        ids = [m.id.replace("models/", "") for m in client.models.list().data]
    except Exception:
        return []
    return sorted(i for i in ids if not any(w in i.lower() for w in skip))[:15]


def _failure_help(provider: str, tried: list, last: "_Exhausted") -> str:
    import openai

    models = ", ".join(tried)
    err = last.error
    if isinstance(err, openai.RateLimitError):
        if last.daily:
            return (
                f"{provider}: the free DAILY quota is used up (models tried: {models}). "
                "It resets later today; or switch provider with LLM_PROVIDER in .env."
            )
        return _rate_limit_help(provider) + f" (models tried: {models})"
    if isinstance(err, openai.InternalServerError):
        return (
            f"{provider}'s servers are overloaded or unavailable (HTTP {err.status_code}) "
            f"(models tried: {models}). This is on the provider's side and usually clears "
            "within minutes. Try again shortly, or switch provider with LLM_PROVIDER in .env."
        )
    return (
        f"Could not reach {provider} after several attempts (models tried: {models}). "
        f"Underlying error: {_cause_text(err)}. Check your internet connection, VPN or "
        "firewall (college/office networks sometimes block API sites), or try another network."
    )


@lru_cache(maxsize=1)
def _openai_class():
    import openai
    from langchain_openai import ChatOpenAI

    class ThrottledChatOpenAI(ChatOpenAI):
        """Spaces out calls, retries with long backoff, and falls back to another
        model of the same provider when the main one keeps failing."""

        def _try_model(self, provider, model, args, kwargs):
            max_attempts = _max_attempts()
            call_kwargs = dict(kwargs)
            if model is not None:
                call_kwargs["model"] = model   # overrides the default model for this call
            for attempt in range(1, max_attempts + 1):
                _throttle()
                try:
                    return super()._generate(*args, **call_kwargs)
                except openai.AuthenticationError as e:
                    raise RuntimeError(_auth_help(provider)) from e
                except openai.RateLimitError as e:
                    if _is_daily_quota(e):
                        raise _Exhausted(e, daily=True) from e
                    err = e
                except openai.InternalServerError as e:   # 500 / 502 / 503 / 504
                    err = e
                except openai.APIConnectionError as e:    # includes timeouts
                    if provider == "ollama":
                        raise RuntimeError(
                            "Cannot reach Ollama. Is it running? Start it with `ollama serve` "
                            "and pull a model, e.g. `ollama pull llama3.2`."
                        ) from e
                    err = e
                if attempt == max_attempts:
                    raise _Exhausted(err) from err
                delay = _backoff(err, attempt)
                detail = _cause_text(err) if isinstance(err, openai.APIConnectionError) else ""
                print(f"{provider} ({model or self.model_name}) is busy "
                      f"[{type(err).__name__}{': ' + detail if detail else ''}]; retrying in "
                      f"{delay:.0f}s (attempt {attempt}/{max_attempts})...")
                time.sleep(delay)

        def _generate(self, *args, **kwargs):
            provider = current_provider()
            primary = self.model_name
            candidates = [primary] + _fallback_models(provider, primary)
            tried, last, not_found = [], None, None
            for i, model in enumerate(candidates):
                if i > 0:
                    print(f"{provider}: switching to fallback model {model} ...")
                try:
                    return self._try_model(provider, None if i == 0 else model, args, kwargs)
                except _Exhausted as e:
                    tried.append(model)
                    last = e
                except openai.NotFoundError as e:
                    not_found = e
                    tried.append(f"{model} (not found)")
                    if i == 0:
                        print(f"{provider}: model '{primary}' was not found (retired?). Update "
                              f"{_PROVIDERS[provider]['model_env']} in .env. Trying fallbacks...")
            if last is None:   # every candidate was "model not found"
                available = _available_models(self.root_client)
                hint = (f" Currently available: {', '.join(available)}." if available
                        else " See the provider's model list.")
                raise RuntimeError(
                    f"{provider} does not have the model(s) {', '.join(tried)}. Models are "
                    f"retired often - set {_PROVIDERS[provider]['model_env']} in .env to a "
                    f"current model name.{hint}"
                ) from not_found
            raise RuntimeError(_failure_help(provider, tried, last)) from last.error

    return ThrottledChatOpenAI


def get_llm(temperature: float = 0.3):
    """The chat model used by every module (selected by LLM_PROVIDER / available key)."""
    provider = current_provider()
    cfg = _PROVIDERS[provider]
    model = _model_name(provider)

    if provider == "mistral":
        key = _api_key("mistral")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not set in .env.\n" + _NO_KEY_HELP)
        return _mistral_class()(
            model=model,
            mistral_api_key=key,
            temperature=temperature,  # must be a float, not a string
        )

    key = _api_key(provider)
    if provider != "ollama" and not key:
        raise RuntimeError(f"{cfg['key_env']} is not set in .env.\n{_NO_KEY_HELP}")

    base_url = cfg["base_url"]
    if provider == "ollama":
        base_url = (os.getenv("OLLAMA_BASE_URL") or "").strip() or base_url

    return _openai_class()(
        model=model,
        base_url=base_url,
        api_key=key or "ollama",      # Ollama ignores the key but the client needs one
        temperature=temperature,
        max_retries=0,   # retries are handled above (longer backoff + fallback model)
        timeout=120,
    )