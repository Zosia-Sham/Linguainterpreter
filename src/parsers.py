from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

def _strip_think_blocks(text) -> str:
    if not text: return ""
    if isinstance(text, list):
        text_parts = []
        for part in text:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(str(part['text']))
            elif isinstance(part, str):
                text_parts.append(part)
        text = "".join(text_parts)
    elif not isinstance(text, str):
        text = str(text)
    return (text.replace("<think>", "").replace("</think>", "")
                .replace("<THINK>", "").replace("</THINK>", ""))

def extract_code(text_or_obj) -> str:
    text = getattr(text_or_obj, "content", text_or_obj) or ""
    t = _strip_think_blocks(text).strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            blocks = [parts[i] for i in range(1, len(parts), 2)]
            block = max(blocks, key=len)
            if block.lstrip().startswith("python"):
                block = block.lstrip()[6:].lstrip()
            return block.strip()
    return t

def extract_boolean(text_or_obj) -> bool:
    t = _strip_think_blocks(getattr(text_or_obj, "content", text_or_obj) or "").strip().lower()
    if "true" in t and "false" not in t: return True
    if "false" in t and "true" not in t: return False
    t0 = t.split()[0] if t.split() else t
    return t0 in ("true", "yes", "pass")

def extract_numbered_list(text_or_obj) -> List[str]:
    txt = _strip_think_blocks(getattr(text_or_obj, "content", text_or_obj) or "")
    out = []
    for l in [x.strip() for x in txt.splitlines() if x.strip()]:
        if l[0].isdigit():
            i = 0
            while i < len(l) and l[i].isdigit(): i += 1
            while i < len(l) and l[i] in ('.', ')', ' '): i += 1
            out.append(l[i:].strip())
        else:
            out.append(l)
    return out

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    text = _strip_think_blocks(text)
    starts = [i for i, ch in enumerate(text) if ch == '{']
    best, best_len = None, -1
    for s in starts:
        depth = 0
        for e in range(s, len(text)):
            if text[e] == '{': depth += 1
            elif text[e] == '}':
                depth -= 1
                if depth == 0:
                    cand = text[s:e+1]
                    if len(cand) > best_len:
                        best, best_len = cand, len(cand)
                    break
    if best:
        try:
            return json.loads(best)
        except Exception:
            return None
    return None
