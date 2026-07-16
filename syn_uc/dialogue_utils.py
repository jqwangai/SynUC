"""Dialogue normalization, prompt rendering, and JSON parsing helpers."""

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ASSISTANT_TRUNCATE_HEAD_CHARS,
    ASSISTANT_TRUNCATE_MIN_CHARS,
    ASSISTANT_TRUNCATE_TAIL_CHARS,
    TRUNCATE_ASSISTANT_IN_PROMPTS,
)


def render_prompt(template: str, **kwargs: str) -> str:
    """替换提示词模板中的占位符。"""
    prompt = template
    for key, value in kwargs.items():
        prompt = prompt.replace(f"<<{key}>>", value)
    return prompt


def messages_to_text(messages: List[Dict[str, str]]) -> str:
    """将 messages 转换为便于 LLM 阅读的对话文本。"""
    lines: List[str] = []
    user_count = 0
    assistant_count = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            user_count += 1
            prefix = f"U{user_count}"
        elif role == "assistant":
            assistant_count += 1
            prefix = f"A{assistant_count}"
            if TRUNCATE_ASSISTANT_IN_PROMPTS and len(content) > ASSISTANT_TRUNCATE_MIN_CHARS:
                head = content[:ASSISTANT_TRUNCATE_HEAD_CHARS]
                tail = (
                    content[-ASSISTANT_TRUNCATE_TAIL_CHARS:]
                    if ASSISTANT_TRUNCATE_TAIL_CHARS > 0
                    else ""
                )
                omitted_chars = len(content) - len(head) - len(tail)
                if omitted_chars > 0:
                    content = (
                        f"{head}……"
                        f"[Intermediate assistant response omitted; only the beginning and ending excerpts are retained; {omitted_chars} characters omitted, original text contains {len(content)} characters in total]"
                        f"……{tail}"
                    )
        elif role == "system":
            prefix = "system"
        else:
            prefix = role or "unknown role"

        lines.append(f"{prefix}：{content}")

    return "\n".join(lines)


def find_last_user_index(messages: List[Dict[str, str]]) -> Optional[int]:
    """找到最后一轮 user 消息的位置。"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def count_user_turns(messages: List[Dict[str, str]]) -> int:
    """统计 user 轮次数。"""
    return sum(1 for msg in messages if msg.get("role") == "user")


def split_for_normal_verification(
    messages: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """把原始样本拆成 history 和最后一轮 user query。"""
    last_user_idx = find_last_user_index(messages)
    if last_user_idx is None:
        return [], None

    history = messages[:last_user_idx]
    current_query = messages[last_user_idx].get("content", "")
    return history, current_query


def normalize_dialogue_item(item: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    """将支持的输入格式统一为 source_id 和 messages。"""
    source_id = str(
        item.get("id")
        or item.get("dialogue_id")
        or item.get("conversation_id")
        or item.get("source_id")
        or f"sample_{random.randint(100000, 999999)}"
    )
    raw_messages = item.get("messages") or item.get("conversation") or []
    messages: List[Dict[str, str]] = []

    if not isinstance(raw_messages, list):
        return source_id, messages

    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"user", "assistant", "system"}:
            continue
        if content is None:
            continue
        messages.append({"role": str(role), "content": str(content)})

    return source_id, messages


def safe_json_dumps(obj: Any) -> str:
    """将对象转成缩进 JSON 字符串，用于放入提示词。"""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def strip_evidence_turn(obj: Any) -> Any:
    """Return a copy with evidence_turn removed from nested dict/list objects."""
    if isinstance(obj, dict):
        return {
            key: strip_evidence_turn(value)
            for key, value in obj.items()
            if key != "evidence_turn"
        }
    if isinstance(obj, list):
        return [strip_evidence_turn(item) for item in obj]
    return obj


def safe_json_dumps_without_evidence_turn(obj: Any) -> str:
    """Dump historical constraints for verifiers without turn-id evidence fields."""
    return safe_json_dumps(strip_evidence_turn(obj))


def extract_json_from_text(text: str) -> Any:
    """从 LLM 输出中提取 JSON 对象。"""
    if text is None:
        return {}

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {}


def compact_candidate_id(value: Any, fallback: int) -> str:
    """把模型输出的 candidate_id 规整成字符串。"""
    candidate_id = str(value).strip() if value is not None else ""
    if not candidate_id:
        candidate_id = str(fallback)
    return candidate_id
