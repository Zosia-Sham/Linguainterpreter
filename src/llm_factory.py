from __future__ import annotations
from typing import Any, Tuple, Optional
import os
import ssl
from .llm_utils import _invoke_with_retry

_ORIGINAL_SSL_CONTEXT_FACTORY = ssl._create_default_https_context

ChatOpenAI = None
ChatAnthropic = None
ChatOllama = None
ChatGoogleGenerativeAI = None
ChatVertexAI = None

def _ensure_google_adc(credentials_path: Optional[str]) -> None:
    if not credentials_path:
        return
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"[ERROR] Google ADC '{credentials_path}' not found.")


def _verify_fast_llm(fast_llm: Any) -> tuple[bool, str]:
    """Minimal invoke: prompt 'hi', any non-empty reply counts as OK."""
    try:
        r = _invoke_with_retry(lambda: fast_llm.invoke("hi"), agent_name="_verify_fast_llm")
        txt = getattr(r, "content", None)
        if txt is None:
            txt = str(r)
        txt = str(txt).strip()
        if not txt:
            return False, "empty response"
        return True, txt[:200]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def build_llms(cfg) -> Tuple[Any, Any, Any]:
    """Возвращает (llm_strong, llm_fast, code_llm) c fallback-логикой."""
    prefer = (cfg.llm.prefer or "openai").lower()
    disable_ssl = bool(getattr(cfg.llm, "disable_ssl", False))
    max_tokens = 32768  # NEW: Set an extremely generous token limit

    if disable_ssl:
        ssl._create_default_https_context = ssl._create_unverified_context
    else:
        ssl._create_default_https_context = _ORIGINAL_SSL_CONTEXT_FACTORY

    def _try_openai():
        global ChatOpenAI
        if ChatOpenAI is None and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI as _ChatOpenAI
                ChatOpenAI = _ChatOpenAI
            except Exception:
                ChatOpenAI = None
        if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
            base_url = cfg.llm.openai.base_url or None
            import httpx
            http_client = httpx.Client(verify=not disable_ssl)
            strong = ChatOpenAI(model=cfg.llm.openai.chat_model_strong, temperature=cfg.llm.openai.temperature, base_url=base_url, http_client=http_client)
            fast   = ChatOpenAI(model=cfg.llm.openai.chat_model_fast,    temperature=cfg.llm.openai.temperature, base_url=base_url, http_client=http_client)
            code   = ChatOpenAI(model=cfg.llm.openai.chat_model_fast,    temperature=min(0.2, cfg.llm.openai.temperature), base_url=base_url, http_client=http_client)
            return strong, fast, code

    def _try_anthropic():
        global ChatAnthropic
        # Lazy re-import: module may be imported before venv site-packages are attached.
        if ChatAnthropic is None and os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic as _ChatAnthropic
                ChatAnthropic = _ChatAnthropic
            except Exception as e:
                print(f"[llm_factory] lazy import langchain_anthropic failed: {e}", flush=True)
                ChatAnthropic = None
        if ChatAnthropic and os.getenv("ANTHROPIC_API_KEY"):
            base_url = cfg.llm.anthropic.base_url or None
            import httpx

            def _anthropic_client(model: str, temperature: float):
                kw: dict = {
                    "model": model,
                    "temperature": temperature,
                    "base_url": base_url,
                    "max_tokens": max_tokens,
                }
                if disable_ssl:
                    kw["http_client"] = httpx.Client(verify=False)
                return ChatAnthropic(**kw)

            strong = _anthropic_client(cfg.llm.anthropic.chat_model_strong, cfg.llm.anthropic.temperature)
            fast = _anthropic_client(cfg.llm.anthropic.chat_model_fast, cfg.llm.anthropic.temperature)
            code = _anthropic_client(
                cfg.llm.anthropic.chat_model_fast,
                min(0.2, cfg.llm.anthropic.temperature),
            )
            return strong, fast, code

    def _try_google():
        global ChatGoogleGenerativeAI
        if ChatGoogleGenerativeAI is None and os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
                ChatGoogleGenerativeAI = _ChatGoogleGenerativeAI
            except Exception:
                ChatGoogleGenerativeAI = None
        if ChatGoogleGenerativeAI and os.getenv("GOOGLE_API_KEY"):
            strong = ChatGoogleGenerativeAI(model=cfg.llm.google.model_pro or "gemini-3.1-pro-preview",
                                            temperature=cfg.llm.google.temperature, max_output_tokens=max_tokens)
            fast   = ChatGoogleGenerativeAI(model=cfg.llm.google.model_flash or "gemini-3-flash-preview",
                                            temperature=cfg.llm.google.temperature, max_output_tokens=max_tokens)
            code   = ChatGoogleGenerativeAI(model=cfg.llm.google.model_pro or "gemini-3.1-pro-preview",
                                            temperature=min(0.1, cfg.llm.google.temperature), max_output_tokens=max_tokens)
            return strong, fast, code

    def _try_vertex():
        global ChatVertexAI
        if ChatVertexAI is None:
            try:
                from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
                ChatVertexAI = _ChatVertexAI
            except Exception:
                ChatVertexAI = None
        if ChatVertexAI:
            creds = cfg.llm.vertex.application_credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
            
            project = cfg.llm.vertex.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = cfg.llm.vertex.location or "us-central1"
            
            if not project and creds and os.path.exists(creds):
                try:
                    with open(creds, 'r') as f:
                        import json
                        data = json.load(f)
                        project = data.get("project_id")
                except Exception:
                    pass

            strong_model = cfg.llm.vertex.model_pro or "gemini-3.1-pro-preview"
            fast_model   = cfg.llm.vertex.model_flash or "gemini-3-flash-preview"

            strong = ChatVertexAI(model=strong_model,  temperature=cfg.llm.vertex.temperature, project=project, location=location, max_output_tokens=max_tokens)
            fast   = ChatVertexAI(model=fast_model,   temperature=cfg.llm.vertex.temperature, project=project, location=location, max_output_tokens=max_tokens)
            code   = ChatVertexAI(model=strong_model,  temperature=min(0.1, cfg.llm.vertex.temperature), project=project, location=location, max_output_tokens=max_tokens)
            return strong, fast, code

    def _try_ollama():
        global ChatOllama
        if ChatOllama is None and cfg.llm.ollama.enabled:
            try:
                from langchain_ollama import ChatOllama as _ChatOllama
                ChatOllama = _ChatOllama
            except Exception:
                ChatOllama = None
        if cfg.llm.ollama.enabled and ChatOllama:
            strong = ChatOllama(model=cfg.llm.ollama.model, temperature=cfg.llm.ollama.temperature, base_url=cfg.llm.ollama.base_url)
            fast   = ChatOllama(model=cfg.llm.ollama.model, temperature=cfg.llm.ollama.temperature, base_url=cfg.llm.ollama.base_url)
            code   = ChatOllama(model=cfg.llm.ollama.model, temperature=cfg.llm.ollama.temperature, base_url=cfg.llm.ollama.base_url)
            return strong, fast, code

    orders = {
        "openai": [_try_openai, _try_anthropic, _try_google, _try_vertex, _try_ollama],
        "anthropic": [_try_anthropic, _try_openai, _try_google, _try_vertex, _try_ollama],
        "google": [_try_google, _try_openai, _try_vertex, _try_ollama],
        "vertex": [_try_vertex, _try_openai, _try_google, _try_ollama],
        "ollama": [_try_ollama, _try_openai, _try_google, _try_vertex],
    }[prefer]

    ping_failures: list[str] = []
    for fn in orders:
        res = fn()
        if not res:
            continue
        strong, fast, code = res
        ok, info = _verify_fast_llm(fast)
        if ok:
            print(f"[llm_factory] backend OK ({fn.__name__}): {info!r}", flush=True)
            return strong, fast, code
        msg = f"{fn.__name__}: {info}"
        ping_failures.append(msg)
        print(f"[llm_factory] backend ping failed — {msg}", flush=True)

    if ping_failures:
        raise RuntimeError(
            "LLM backend(s) built but none responded to ping. "
            + " | ".join(ping_failures)
        )
    # Debug without leaking secrets: only booleans.
    debug = {
        "prefer": prefer,
        "ANTHROPIC_API_KEY_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "ChatAnthropic_imported": globals().get("ChatAnthropic") is not None,
        "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
        "ChatOpenAI_imported": globals().get("ChatOpenAI") is not None,
        "OLLAMA_enabled": bool(getattr(getattr(cfg, "llm", None), "ollama", None) and getattr(cfg.llm.ollama, "enabled", False)),
        "ChatOllama_imported": globals().get("ChatOllama") is not None,
    }
    print("[llm_factory] No backend available debug:", debug, flush=True)
    raise RuntimeError("No LLM backend available. Provide keys or enable Ollama.")
