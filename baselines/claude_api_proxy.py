#!/usr/bin/env python3
# ~/claude-proxy.py
#
# Прокси для Claude Code.
# Claude Code шлёт нативный Anthropic Messages API -> конвертируем в OpenAI Chat Completions.
#
# Запуск:
#   python3 ~/claude-proxy.py
#
# Использование:
#   ANTHROPIC_BASE_URL=http://localhost:3212 ANTHROPIC_API_KEY=any claude

import json
import time
import random
import sys
import os
import ssl
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urljoin

DEBUG = True
MAX_MSG_LENGTH = 500
TRUNCATE_MARKER = " ... [TRUNCATED] ... "

CONFIG_FILE    = os.path.expanduser("~/.claude-proxy-config.json")
MODELS_FILE    = os.path.expanduser("~/working_models_clean.json")
PROXY_PORT     = 3212
MAX_RETRIES    = 30
MAX_BACKOFF    = 300.0
INITIAL_BACKOFF = 2.0
MAX_TOKENS     = 60000

# 413 trimming — upstream лимит явно ниже 242KB судя по логам
MAX_PAYLOAD_BYTES      = 180_000   # агрессивный порог, режем заранее
CONTENT_TRUNCATE_CHARS = 2000
MIN_MESSAGES_KEEP      = 4
TOOL_DESC_TRUNCATE_CHARS = 200     # режем descriptions тулзов сильнее
TOOL_PARAM_TRUNCATE    = True
MAX_TOOLS_KEEP         = 50        # максимум тулзов при emergency trim

# finish_reason -> Anthropic stop_reason
FINISH_REASON_MAP = {
    "stop":         "end_turn",
    "tool_calls":   "tool_use",
    "length":       "max_tokens",
    "content_filter": "stop_sequence",
}


class Colors:
    CYAN = '\033[96m'; YELLOW = '\033[93m'; GREEN = '\033[92m'
    RED = '\033[91m';  GRAY = '\033[90m';   MAGENTA = '\033[95m'
    BLUE = '\033[94m'; END = '\033[0m'


def truncate_middle(text, max_len=MAX_MSG_LENGTH):
    if not text or len(text) <= max_len:
        return text
    half = (max_len - len(TRUNCATE_MARKER)) // 2
    return text[:half] + TRUNCATE_MARKER + text[-half:]


# ---------------------------------------------------------------------------
# Выбор модели (один в один с codex-proxy)
# ---------------------------------------------------------------------------

class ModelSelector:
    def __init__(self, models):
        self.models   = models
        self.filtered = models.copy()
        self.page     = 0
        self.per_page = 15

    def display(self):
        os.system('clear' if os.name != 'nt' else 'cls')
        total = (len(self.filtered) // self.per_page) + (1 if len(self.filtered) % self.per_page else 0)
        print(f"\033[1;96m=== ВЫБОР МОДЕЛИ (claude-proxy) ===\033[0m")
        print(f"Всего: {len(self.models)} | Фильтр: {len(self.filtered)} | Стр. {self.page+1}/{total}")
        print("-" * 80)
        start = self.page * self.per_page
        for i, m in enumerate(self.filtered[start:start+self.per_page], start=start+1):
            ep     = m['endpoint'].replace('https://', '').replace('/v1', '')[:25]
            marker = "\033[92m★\033[0m" if m['model'].startswith(('azure/', 'openai/gpt-4')) else " "
            print(f"{marker} {i:3}. \033[96m{m['model']:45}\033[0m ({ep}...)")
        print("-" * 80)
        print(f"\033[93mn\033[0m-вперед \033[93mp\033[0m-назад \033[93ms\033[0m-поиск \033[93mq\033[0m-выход")

    def search(self, query):
        query = query.lower()
        self.filtered = [m for m in self.models if query in m['model'].lower()]
        self.page = 0
        if not self.filtered:
            self.filtered = self.models.copy()

    def run(self):
        while True:
            self.display()
            cmd = input("\n> ").strip().lower()
            if cmd == 'q':
                sys.exit(0)
            elif cmd == 'n':
                if (self.page + 1) * self.per_page < len(self.filtered):
                    self.page += 1
            elif cmd == 'p':
                if self.page > 0:
                    self.page -= 1
            elif cmd.startswith('s '):
                self.search(cmd[2:])
            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(self.filtered):
                    return self.filtered[idx]


# ---------------------------------------------------------------------------
# HTTP-обработчик
# ---------------------------------------------------------------------------

class _ClientDisconnected(Exception):
    """Клиент закрыл соединение до конца стрима."""


class ClaudeProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        # /v1/models — отдаём фейковый список чтобы клиенты не падали
        if self.path.startswith("/v1/models"):
            model = self.server.config["model"]
            body = json.dumps({
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "created": int(time.time()), "owned_by": "proxy"},
                    # Claude Code часто запрашивает конкретные имена — добавляем алиасы
                    {"id": "claude-opus-4-6",        "object": "model", "created": int(time.time()), "owned_by": "proxy"},
                    {"id": "claude-sonnet-4-6",      "object": "model", "created": int(time.time()), "owned_by": "proxy"},
                    {"id": "claude-haiku-4-5-20251001","object": "model","created": int(time.time()), "owned_by": "proxy"},
                ]
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        self._respond_raw({"status": 404, "body": b'{"error":"not found"}'})

    def do_POST(self):
        try:
            self._process_post()
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[{time.strftime('%H:%M:%S')}] Client disconnect: {e}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------

    def _process_post(self):
        session_id = self.headers.get("X-Claude-Code-Session-Id", "unknown")
        self.request.settimeout(600)

        cl   = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(cl).decode('utf-8') if cl > 0 else None
        data = json.loads(body) if body else {}

        original_model = data.get("model", "unknown")
        data["model"]  = self.server.config["model"]

        is_streaming = data.get("stream", False)

        print(f"[{time.strftime('%H:%M:%S')}] {'='*60}")
        print(f"[{time.strftime('%H:%M:%S')}] POST {self.path}")
        print(f"[{time.strftime('%H:%M:%S')}] Model: {original_model} -> {data['model']}")
        print(f"[{time.strftime('%H:%M:%S')}] Streaming: {is_streaming}")
        print(f"[{time.strftime('%H:%M:%S')}] Messages: {len(data.get('messages', []))}, Tools: {len(data.get('tools', []))}")

        if "/v1/messages/count_tokens" in self.path:
            self._handle_count_tokens(data)
            return

        if "/v1/messages" in self.path:
            chat_data = self._anthropic_to_openai(data)
            url       = f"{self.server.config['endpoint']}/chat/completions"

            if is_streaming:
                self._handle_streaming(url, chat_data, data, session_id)
            else:
                self._handle_blocking(url, chat_data, data, session_id)
            return

        # Всё остальное — проксируем как есть
        url    = urljoin(self.server.config['endpoint'], self.path)
        result = self._upstream(url, "POST", data, False)
        self._respond_raw(result)

    # ------------------------------------------------------------------
    # count_tokens — возвращаем заглушку
    # ------------------------------------------------------------------

    def _handle_count_tokens(self, data):
        # Грубая оценка: ~4 символа на токен
        total = 0
        for m in data.get("messages", []):
            c = m.get("content", "")
            if isinstance(c, str):
                total += len(c) // 4
            elif isinstance(c, list):
                for b in c:
                    total += len(str(b.get("text", "") or b.get("input", ""))) // 4
        resp = json.dumps({"input_tokens": max(1, total)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(resp)

    # ------------------------------------------------------------------
    # Blocking (non-streaming)
    # ------------------------------------------------------------------

    def _handle_blocking(self, url, chat_data, original_data, session_id):
        chat_data, _ = self._trim_payload(chat_data)

        for attempt in range(MAX_RETRIES):
            result = self._upstream(url, "POST", chat_data, False)
            status = result.get("status", 0)

            if status == 200:
                anthropic_resp = self._openai_to_anthropic(result, original_data, session_id)
                self._respond_raw(anthropic_resp)
                return

            if status == 413:
                chat_data = self._emergency_trim(chat_data, attempt)
                continue

            if status == 429:
                wait = self._backoff(attempt)
                print(f"[{time.strftime('%H:%M:%S')}] 429, retry {attempt+1}/{MAX_RETRIES}, wait {wait:.1f}s")
                time.sleep(wait)
                continue

            if status == 400:
                err = result.get("body", b"").decode()
                print(f"[{time.strftime('%H:%M:%S')}] 400: {err[:300]}")
                # Пробуем переключить max_tokens / max_completion_tokens
                if "max_completion_tokens" in err and "max_tokens" in chat_data:
                    chat_data["max_completion_tokens"] = chat_data.pop("max_tokens")
                    continue
                if "temperature" in err and "temperature" in chat_data:
                    del chat_data["temperature"]
                    continue
                break

            if status >= 500 and attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** (attempt // 3), 30))
                continue

            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            break

        print(f"[{time.strftime('%H:%M:%S')}] Failed, last status: {result.get('status')}")
        # Возвращаем ошибку в формате Anthropic
        err_body = json.dumps({
            "type": "error",
            "error": {"type": "api_error", "message": f"Upstream error {result.get('status')}: {result.get('body', b'').decode()[:200]}"}
        }).encode()
        self._respond_raw({"status": result.get("status", 500), "body": err_body})

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _handle_streaming(self, url, chat_data, original_data, session_id):
        # Запрашиваем у апстрима без стриминга — синтезируем SSE сами
        chat_data["stream"] = False
        chat_data, _ = self._trim_payload(chat_data)

        result = None
        for attempt in range(MAX_RETRIES):
            result = self._upstream(url, "POST", chat_data, False)
            status = result.get("status", 0)

            if status == 200:
                self._emit_anthropic_sse(result, original_data, session_id)
                return

            if status == 413:
                chat_data = self._emergency_trim(chat_data, attempt)
                continue

            if status == 429:
                wait = self._backoff(attempt)
                print(f"[{time.strftime('%H:%M:%S')}] 429, retry {attempt+1}/{MAX_RETRIES}, wait {wait:.1f}s")
                time.sleep(wait)
                continue

            if status == 400:
                err = result.get("body", b"").decode()
                if "max_completion_tokens" in err and "max_tokens" in chat_data:
                    chat_data["max_completion_tokens"] = chat_data.pop("max_tokens")
                    continue
                if "temperature" in err and "temperature" in chat_data:
                    del chat_data["temperature"]
                    continue
                break

            if status >= 500 and attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** (attempt // 3), 30))
                continue

            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            break

        # Ошибка в SSE-формате
        self._emit_sse_error(result)

    def _emit_anthropic_sse(self, result, original_data, session_id):
        """Конвертируем OpenAI-ответ в Anthropic SSE поток."""
        try:
            r       = json.loads(result["body"].decode())
            choice  = r["choices"][0]
            message = choice.get("message", {})
            text    = message.get("content") or ""
            tcs     = message.get("tool_calls") or []

            finish  = choice.get("finish_reason", "stop")
            stop_reason = FINISH_REASON_MAP.get(finish, "end_turn")

            msg_id     = r.get("id", f"msg_{int(time.time())}")
            model_name = r.get("model", original_data.get("model", "unknown"))
            created_at = r.get("created", int(time.time()))

            upstream_usage = r.get("usage", {})
            
            # TOKEN STATS FOR EXPRMNTS
            with open("usage.log", "a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "session": session_id,
                    "requested_model": original_data.get("model"),
                    "actual_model": r.get("model"),
                    "prompt_tokens": upstream_usage.get("prompt_tokens", 0),
                    "completion_tokens": upstream_usage.get("completion_tokens", 0),
                    "total_tokens": upstream_usage.get("total_tokens", 0),
                }) + "\n")
            
            input_tok  = upstream_usage.get("prompt_tokens", 0)
            output_tok = upstream_usage.get("completion_tokens", 0)

            print(f"[{time.strftime('%H:%M:%S')}] [SSE] text={len(text)}, tool_calls={len(tcs)}, stop={stop_reason}")

            # --- Открываем стрим ---
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # 1. message_start
            self._sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": model_name,
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": input_tok, "output_tokens": 1}
                }
            })

            self._sse("ping", {"type": "ping"})

            block_idx = 0

            # 2. Текстовый блок
            if text:
                self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""}
                })
                # Отдаём текст чанками по ~100 символов для реалистичного стриминга
                chunk_size = 100
                for i in range(0, len(text), chunk_size):
                    self._sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": text[i:i+chunk_size]}
                    })
                self._sse("content_block_stop", {
                    "type": "content_block_stop", "index": block_idx
                })
                block_idx += 1

            # 3. tool_use блоки
            for tc in tcs:
                func    = tc.get("function", {})
                tc_id   = tc.get("id", f"toolu_{int(time.time())}")
                tc_name = func.get("name", "unknown")
                tc_args = func.get("arguments", "{}")
                print(f"[{time.strftime('%H:%M:%S')}]   tool_use: {tc_name}")

                self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "tool_use", "id": tc_id, "name": tc_name, "input": {}}
                })
                # Отдаём JSON аргументов кусками
                chunk_size = 64
                for i in range(0, len(tc_args), chunk_size):
                    self._sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "input_json_delta", "partial_json": tc_args[i:i+chunk_size]}
                    })
                self._sse("content_block_stop", {
                    "type": "content_block_stop", "index": block_idx
                })
                block_idx += 1

            # 4. message_delta
            self._sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tok}
            })

            # 5. message_stop
            self._sse("message_stop", {"type": "message_stop"})

            # Явно закрываем соединение — без этого клиент висит в ожидании
            try:
                self.wfile.flush()
                self.connection.shutdown(1)  # SHUT_WR
            except Exception:
                pass

            print(f"[{time.strftime('%H:%M:%S')}] [SSE DONE] {block_idx} blocks")

        except _ClientDisconnected:
            print(f"[{time.strftime('%H:%M:%S')}] [SSE] Client disconnected mid-stream")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [SSE ERROR] {e}")
            import traceback; traceback.print_exc()

    def _emit_sse_error(self, result):
        status = result.get("status", 500) if result else 500
        msg    = result.get("body", b"").decode()[:200] if result else "unknown error"
        try:
            self.send_response(200)  # SSE всегда 200, ошибка внутри потока
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self._sse("error", {
                "type": "error",
                "error": {"type": "api_error", "message": f"Upstream {status}: {msg}"}
            })
        except Exception:
            pass

    def _sse(self, event_name, data):
        line = f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        try:
            self.wfile.write(line.encode('utf-8'))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise _ClientDisconnected()

    # ------------------------------------------------------------------
    # Конвертация Anthropic Messages -> OpenAI Chat Completions
    # ------------------------------------------------------------------

    def _anthropic_to_openai(self, data):
        """
        Конвертируем Anthropic Messages API запрос в OpenAI Chat Completions.

        Обрабатываем все типы content-блоков:
          text / tool_use / tool_result / image (base64 и url)
        """
        model_lower = data.get("model", "").lower()
        is_new = any(x in model_lower for x in [
            "gpt-5", "o1", "o3", "o4", "claude-3-5", "claude-4", "gemini-2", "deepseek"
        ])

        out = {"model": data.get("model"), "messages": []}

        if is_new:
            out["max_completion_tokens"] = data.get("max_tokens", MAX_TOKENS)
        else:
            out["max_tokens"] = data.get("max_tokens", MAX_TOKENS)

        if "temperature" in data:
            out["temperature"] = data["temperature"]

        out["stream"] = False  # стримим сами

        # system
        system = data.get("system")
        if system:
            if isinstance(system, str):
                out["messages"].append({"role": "system", "content": system})
            elif isinstance(system, list):
                # system как массив блоков (редко, но бывает)
                text_parts = [b.get("text", "") for b in system if b.get("type") == "text"]
                out["messages"].append({"role": "system", "content": "\n".join(text_parts)})

        # messages
        for msg in data.get("messages", []):
            role    = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                out["messages"].append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                out["messages"].append({"role": role, "content": str(content)})
                continue

            # Разбираем content-блоки
            # Сначала собираем tool_result блоки — они становятся отдельными tool-сообщениями
            tool_result_msgs = []
            main_parts       = []
            tool_calls_list  = []

            for block in content:
                btype = block.get("type", "")

                if btype == "text":
                    main_parts.append({"type": "text", "text": block.get("text", "")})

                elif btype == "tool_use":
                    # Вызов инструмента ассистентом
                    tc_id   = block.get("id", f"call_{int(time.time())}")
                    tc_name = block.get("name", "unknown")
                    tc_args = block.get("input", {})
                    if isinstance(tc_args, dict):
                        tc_args = json.dumps(tc_args, ensure_ascii=False)
                    elif not isinstance(tc_args, str):
                        tc_args = str(tc_args)
                    tool_calls_list.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": tc_name, "arguments": tc_args}
                    })

                elif btype == "tool_result":
                    # Результат инструмента — отдельное сообщение role=tool
                    tc_id  = block.get("tool_use_id", f"call_{int(time.time())}")
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        # content может быть массивом текстовых блоков
                        result_content = "\n".join(
                            b.get("text", "") for b in result_content if b.get("type") == "text"
                        )
                    elif not isinstance(result_content, str):
                        result_content = str(result_content)
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result_content
                    })

                elif btype == "image":
                    # Изображение
                    src = block.get("source", {})
                    src_type = src.get("type", "")
                    if src_type == "base64":
                        media = src.get("media_type", "image/jpeg")
                        b64   = src.get("data", "")
                        main_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media};base64,{b64}"}
                        })
                    elif src_type == "url":
                        main_parts.append({
                            "type": "image_url",
                            "image_url": {"url": src.get("url", "")}
                        })

                else:
                    # Неизвестный тип — пробуем вытащить текст
                    if "text" in block:
                        main_parts.append({"type": "text", "text": block["text"]})

            # Собираем основное сообщение.
            # Важно: tool_result должны идти ДО text-части того же user-сообщения,
            # потому что OpenAI требует assistant(tool_calls) -> tool(result) без разрывов.
            if tool_calls_list:
                # assistant с tool_calls
                text_content = None
                if main_parts:
                    if len(main_parts) == 1 and main_parts[0]["type"] == "text":
                        text_content = main_parts[0]["text"]
                    else:
                        text_content = main_parts
                out["messages"].append({
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": tool_calls_list
                })
                # tool_result после assistant — правильный порядок
                out["messages"].extend(tool_result_msgs)
            else:
                # Сначала tool_results (закрывают предыдущий assistant+tool_calls)
                out["messages"].extend(tool_result_msgs)
                # Потом текстовая часть этого же user-сообщения
                if main_parts:
                    if len(main_parts) == 1 and main_parts[0]["type"] == "text":
                        out["messages"].append({"role": role, "content": main_parts[0]["text"]})
                    else:
                        out["messages"].append({"role": role, "content": main_parts})

        # tools: Anthropic использует input_schema вместо parameters
        if "tools" in data:
            oai_tools = []
            for t in data["tools"]:
                if not isinstance(t, dict):
                    continue
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {}))
                    }
                })
            if oai_tools:
                out["tools"] = oai_tools

        # tool_choice
        if "tool_choice" in data:
            tc = data["tool_choice"]
            if isinstance(tc, dict):
                tc_type = tc.get("type", "auto")
                if tc_type == "auto":
                    out["tool_choice"] = "auto"
                elif tc_type == "any":
                    out["tool_choice"] = "required"
                elif tc_type == "tool":
                    out["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}
                elif tc_type == "none":
                    out["tool_choice"] = "none"
            elif isinstance(tc, str):
                out["tool_choice"] = tc

        msg_count  = len(out["messages"])
        tool_count = len(out.get("tools", []))
        max_tok    = out.get("max_completion_tokens") or out.get("max_tokens")
        roles      = [m.get("role", "?") + ("(tc)" if m.get("tool_calls") else "") for m in out["messages"]]
        print(f"[{time.strftime('%H:%M:%S')}] -> OAI: {msg_count} msgs, {tool_count} tools, max_tokens={max_tok}")
        print(f"[{time.strftime('%H:%M:%S')}]    roles: {roles}")

        return out

    # ------------------------------------------------------------------
    # Конвертация OpenAI Chat Completions -> Anthropic Messages (не-стрим)
    # ------------------------------------------------------------------

    def _openai_to_anthropic(self, result, original_data, session_id):
        """Конвертируем OpenAI-ответ обратно в Anthropic Messages формат."""
        try:
            r      = json.loads(result["body"].decode())
            choice = r["choices"][0]
            msg    = choice.get("message", {})
            text   = msg.get("content") or ""
            tcs    = msg.get("tool_calls") or []

            finish      = choice.get("finish_reason", "stop")
            stop_reason = FINISH_REASON_MAP.get(finish, "end_turn")

            content = []
            if text:
                content.append({"type": "text", "text": text})

            for tc in tcs:
                func    = tc.get("function", {})
                tc_id   = tc.get("id", f"toolu_{int(time.time())}")
                tc_name = func.get("name", "")
                tc_args = func.get("arguments", "{}")
                try:
                    parsed = json.loads(tc_args)
                except Exception:
                    parsed = tc_args
                content.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": tc_name,
                    "input": parsed
                })

            upstream_usage = r.get("usage", {})
            
            # TOKEN STATS FOR EXPRMNTS
            with open("usage.log", "a") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "session": session_id,
                    "requested_model": original_data.get("model"),
                    "actual_model": r.get("model"),
                    "prompt_tokens": upstream_usage.get("prompt_tokens", 0),
                    "completion_tokens": upstream_usage.get("completion_tokens", 0),
                    "total_tokens": upstream_usage.get("total_tokens", 0),
                }) + "\n")
            
            anthropic_resp = {
                "id":            r.get("id", f"msg_{int(time.time())}"),
                "type":          "message",
                "role":          "assistant",
                "content":       content,
                "model":         original_data.get("model", r.get("model", "unknown")),
                "stop_reason":   stop_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens":  upstream_usage.get("prompt_tokens", 0),
                    "output_tokens": upstream_usage.get("completion_tokens", 0),
                }
            }

            print(f"[{time.strftime('%H:%M:%S')}] -> Anthropic: text={len(text)}, tool_use={len(tcs)}, stop={stop_reason}")
            body = json.dumps(anthropic_resp, ensure_ascii=False).encode()
            return {"status": 200, "body": body}

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] _openai_to_anthropic error: {e}")
            import traceback; traceback.print_exc()
            err = json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}}).encode()
            return {"status": 500, "body": err}

    # ------------------------------------------------------------------
    # Upstream HTTP
    # ------------------------------------------------------------------

    def _upstream(self, url, method, data, streaming=False):
        STRIP_HEADERS = {
            'host', 'content-length', 'origin',
            'authorization', 'x-api-key',
            'anthropic-version', 'anthropic-beta',
            'x-stainless-os', 'x-stainless-lang', 'x-stainless-runtime',
            'x-stainless-runtime-version', 'x-stainless-package-version',
            'x-stainless-arch', 'x-stainless-async',
        }
        body    = json.dumps(data, ensure_ascii=False).encode() if data else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in STRIP_HEADERS}
        headers["Authorization"] = f"Bearer {self.server.config['api_key']}"
        headers["Content-Type"]  = "application/json"

        print(f"[{time.strftime('%H:%M:%S')}] [UPSTREAM] {method} {url} ({len(body or b'')} bytes)")

        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx, timeout=300) as r:
                resp_body = r.read()
                print(f"[{time.strftime('%H:%M:%S')}] [UPSTREAM] {r.status} OK ({len(resp_body)} bytes)")
                return {"status": r.status, "headers": dict(r.headers), "body": resp_body}
        except urllib.error.HTTPError as e:
            err_body = e.read()
            print(f"[{time.strftime('%H:%M:%S')}] [UPSTREAM] {e.code} ERR: {err_body[:300]}")
            return {"status": e.code, "headers": dict(e.headers), "body": err_body}
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [UPSTREAM] EXCEPTION: {e}")
            return {"status": 0, "error": str(e), "body": b""}

    def _respond_raw(self, result):
        try:
            self.send_response(result.get("status", 500))
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            if "body" in result:
                self.send_header("Content-Length", len(result["body"]))
            self.end_headers()
            if "body" in result:
                self.wfile.write(result["body"])
                self.wfile.flush()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] _respond_raw error: {e}")

    # ------------------------------------------------------------------
    # 413 trimming (та же логика что в codex-proxy)
    # ------------------------------------------------------------------

    def _payload_size(self, d):
        return len(json.dumps(d, ensure_ascii=False).encode())

    def _truncate_content(self, content, max_chars):
        if isinstance(content, str):
            if len(content) > max_chars:
                half = max_chars // 2
                return content[:half] + f" ... [~{len(content)} chars, truncated] ... " + content[-half:]
            return content
        if isinstance(content, list):
            return [
                {"type": "text", "text": self._truncate_content(p["text"], max_chars)}
                if isinstance(p, dict) and p.get("type") == "text" else p
                for p in content
            ]
        return content

    def _trim_payload(self, chat_data):
        import copy
        d    = copy.deepcopy(chat_data)
        msgs = d.get("messages", [])

        if self._payload_size(d) <= MAX_PAYLOAD_BYTES:
            return d, False

        original_size = self._payload_size(chat_data)

        # Шаг 0: сразу режем количество тулзов — самый жирный вклад в размер
        if "tools" in d and len(d["tools"]) > MAX_TOOLS_KEEP:
            before = len(d["tools"])
            d["tools"] = d["tools"][:MAX_TOOLS_KEEP]
            print(f"[{time.strftime('%H:%M:%S')}] [TRIM] tools {before} -> {MAX_TOOLS_KEEP} (step 0)")

        changed = True
        rounds  = 0

        while self._payload_size(d) > MAX_PAYLOAD_BYTES and changed and rounds < 20:
            changed = False
            rounds += 1
            tail      = MIN_MESSAGES_KEEP
            mid_range = range(1, max(1, len(msgs) - tail))

            # Шаг 1: обрезаем content у средних сообщений
            for i in mid_range:
                m = msgs[i]
                c = m.get("content")
                if c:
                    new_c = self._truncate_content(c, CONTENT_TRUNCATE_CHARS)
                    if new_c != c:
                        msgs[i] = dict(m, content=new_c)
                        changed = True

            if self._payload_size(d) <= MAX_PAYLOAD_BYTES:
                break

            # Шаг 2: дропаем средние группы целиком (assistant+tool_calls вместе с tool-результатами)
            if len(msgs) > MIN_MESSAGES_KEEP + 2:
                # Находим первую "средную" группу которую можно дропнуть
                # (не system[0], не последние MIN_MESSAGES_KEEP)
                i = 1
                while i < max(1, len(msgs) - tail):
                    m = msgs[i]
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        # Найти конец группы (все следующие tool)
                        j = i + 1
                        while j < len(msgs) and msgs[j].get("role") == "tool":
                            j += 1
                        del msgs[i:j]
                        d["messages"] = msgs
                        changed = True
                        break
                    else:
                        # Одиночное сообщение — дропаем его
                        del msgs[i]
                        d["messages"] = msgs
                        changed = True
                        break

            if self._payload_size(d) <= MAX_PAYLOAD_BYTES:
                break

            # Шаг 3: режем descriptions у tools
            if "tools" in d:
                for t in d["tools"]:
                    fn   = t.get("function", {})
                    desc = fn.get("description", "")
                    if len(desc) > TOOL_DESC_TRUNCATE_CHARS:
                        fn["description"] = desc[:TOOL_DESC_TRUNCATE_CHARS] + "..."
                        changed = True

            if self._payload_size(d) <= MAX_PAYLOAD_BYTES:
                break

            # Шаг 4: убираем properties у parameters
            if TOOL_PARAM_TRUNCATE and "tools" in d:
                for t in d["tools"]:
                    fn     = t.get("function", {})
                    params = fn.get("parameters", {})
                    if "properties" in params and params["properties"]:
                        req = params.get("required", [])
                        fn["parameters"] = {
                            "type": params.get("type", "object"),
                            "properties": {k: v for k, v in params["properties"].items() if k in req},
                            "required": req
                        }
                        changed = True

            if self._payload_size(d) <= MAX_PAYLOAD_BYTES:
                break

            # Шаг 5: дропаем лишние тулзы (оставляем MAX_TOOLS_KEEP)
            if "tools" in d and len(d["tools"]) > MAX_TOOLS_KEEP:
                before = len(d["tools"])
                d["tools"] = d["tools"][:MAX_TOOLS_KEEP]
                print(f"[{time.strftime('%H:%M:%S')}] [TRIM] tools {before} -> {len(d['tools'])}")
                changed = True

        final = self._payload_size(d)
        was_changed = final != original_size
        if was_changed:
            print(f"[{time.strftime('%H:%M:%S')}] [TRIM] {original_size//1024}KB -> {final//1024}KB, "
                  f"msgs: {len(chat_data.get('messages',[]))} -> {len(msgs)}, "
                  f"tools: {len(chat_data.get('tools',[]))} -> {len(d.get('tools',[]))}")
        return d, was_changed

    def _emergency_trim(self, chat_data, attempt):
        """При 413 — режем агрессивно, но никогда не разрываем tool_calls/tool_result пары."""
        import copy
        tmp      = copy.deepcopy(chat_data)
        tmp_msgs = tmp.get("messages", [])

        # Группируем сообщения в атомарные блоки которые нельзя разрывать:
        #   [system], [user], [assistant+tool_calls, tool, tool, ...], [user], ...
        groups = []
        i = 0
        while i < len(tmp_msgs):
            m = tmp_msgs[i]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                # Собираем всю группу: assistant + все следующие tool-сообщения
                group = [m]
                j = i + 1
                while j < len(tmp_msgs) and tmp_msgs[j].get("role") == "tool":
                    group.append(tmp_msgs[j])
                    j += 1
                groups.append(group)
                i = j
            else:
                groups.append([m])
                i += 1

        # Оставляем: первую группу (system/user) + последние N групп
        # Каждый attempt берём меньше
        max_groups = max(2, len(groups) - attempt - 1)
        if len(groups) > max_groups:
            kept = groups[:1] + groups[-(max_groups - 1):]
            tmp["messages"] = [m for g in kept for m in g]
            print(f"[{time.strftime('%H:%M:%S')}] [413 TRIM] msgs {len(tmp_msgs)} -> {len(tmp['messages'])} "
                  f"(groups {len(groups)} -> {len(kept)})")

        # Режем тулзы: каждый attempt берём меньше
        if "tools" in tmp and tmp["tools"]:
            max_tools = max(10, MAX_TOOLS_KEEP - attempt * 20)
            if len(tmp["tools"]) > max_tools:
                before = len(tmp["tools"])
                tmp["tools"] = tmp["tools"][:max_tools]
                print(f"[{time.strftime('%H:%M:%S')}] [413 TRIM] tools {before} -> {max_tools}")

            # Зануляем parameters и режем descriptions
            for t in tmp["tools"]:
                fn = t.get("function", {})
                fn["description"] = fn.get("description", "")[:100]
                fn["parameters"]  = {"type": "object", "properties": {}, "required": []}

        new_size = self._payload_size(tmp)
        print(f"[{time.strftime('%H:%M:%S')}] [413 TRIM] -> {new_size//1024}KB")
        return tmp

    def _backoff(self, attempt):
        return min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF) + random.uniform(0, 1)


# ---------------------------------------------------------------------------
# ProxyServer
# ---------------------------------------------------------------------------

class ClaudeProxyServer(HTTPServer):
    def __init__(self, addr, handler, models_data):
        super().__init__(addr, handler)
        self.models_data = models_data
        self.config      = self._load_or_select()

    def _load_or_select(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            if any(m["model"] == saved["model"] for m in self.models_data.get("working", [])):
                print(f"\033[92mПоследняя: {saved['model']}\033[0m")
                if input("Использовать? (y/n): ").strip().lower() == 'y':
                    return saved

        selector = ModelSelector(self.models_data.get("working", []))
        selected = selector.run()

        ep = selected["endpoint"]
        if not ep.endswith("/v1"):
            ep = ep.rstrip("/") + "/v1"

        config = {
            "model":    selected["model"],
            "endpoint": ep,
            "api_key":  "key"
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        print(f"\033[92m✓ {config['model']}\033[0m")
        return config


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if "--reset" in sys.argv and os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
        print(f"\033[93mСохранённый конфиг удалён, выбери модель заново\033[0m")

    if not os.path.exists(MODELS_FILE):
        print(f"Нет {MODELS_FILE}")
        sys.exit(1)

    with open(MODELS_FILE) as f:
        data = json.load(f)

    server = ClaudeProxyServer(("localhost", PROXY_PORT), ClaudeProxyHandler, data)

    print(f"\n\033[92m✓ claude-proxy localhost:{PROXY_PORT}\033[0m")
    print(f"\033[95m  MAX_TOKENS:  {MAX_TOKENS:,}\033[0m")
    print(f"\033[95m  MAX_RETRIES: {MAX_RETRIES}\033[0m")
    print(f"  Модель:   {server.config['model']}")
    print(f"  Endpoint: {server.config['endpoint']}")
    print(f"\n\033[93m── Запуск Claude Code ──\033[0m")
    print(f"  # Если видишь 'Auth conflict' — сначала:")
    print(f"  claude /logout")
    print(f"")
    print(f"  # Затем:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT} ANTHROPIC_API_KEY=proxy claude")
    print(f"\n  # Или одной строкой (без logout, если нет конфликта):")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PROXY_PORT} ANTHROPIC_API_KEY=proxy claude\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nПока")


if __name__ == "__main__":
    main()
