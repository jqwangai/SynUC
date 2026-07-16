"""Thread-aware token usage accounting."""

import json
import threading
from typing import Any, Dict, Optional


TOKEN_USAGE: Dict[str, Any] = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "by_operation": {},
}
TOKEN_USAGE_LOCK = threading.Lock()
THREAD_STATE = threading.local()


def extract_token_usage(response: Any) -> Dict[str, int]:
    """从一次 LLM 响应中提取 token usage。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def new_token_usage() -> Dict[str, Any]:
    """Create an empty token usage accumulator."""
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_operation": {},
    }


def get_thread_token_usage() -> Dict[str, Any]:
    """Return the current worker thread's token usage accumulator."""
    usage = getattr(THREAD_STATE, "token_usage", None)
    if usage is None:
        usage = new_token_usage()
        THREAD_STATE.token_usage = usage
    return usage


def add_single_call_usage(
    target: Dict[str, Any],
    operation: str,
    usage: Dict[str, int],
) -> None:
    """Add one LLM call's usage into an accumulator."""
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    target["calls"] += 1
    target["prompt_tokens"] += prompt_tokens
    target["completion_tokens"] += completion_tokens
    target["total_tokens"] += total_tokens

    by_operation = target["by_operation"]
    if operation not in by_operation:
        by_operation[operation] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    op_usage = by_operation[operation]
    op_usage["calls"] += 1
    op_usage["prompt_tokens"] += prompt_tokens
    op_usage["completion_tokens"] += completion_tokens
    op_usage["total_tokens"] += total_tokens


def record_token_usage(operation: str, response: Any) -> Dict[str, int]:
    """累计一次 LLM 响应中的 token usage。"""
    usage = extract_token_usage(response)
    add_single_call_usage(get_thread_token_usage(), operation, usage)
    with TOKEN_USAGE_LOCK:
        add_single_call_usage(TOKEN_USAGE, operation, usage)
    return usage


def reset_token_usage() -> None:
    """重置当前运行的 token usage 统计。"""
    THREAD_STATE.token_usage = new_token_usage()


def snapshot_token_usage() -> Dict[str, Any]:
    """复制当前 token usage 统计，用于后续计算样本级消耗。"""
    return json.loads(json.dumps(get_thread_token_usage(), ensure_ascii=False))


def reset_all_token_usage() -> None:
    """Reset current-thread and run-level token usage."""
    reset_token_usage()
    with TOKEN_USAGE_LOCK:
        TOKEN_USAGE["calls"] = 0
        TOKEN_USAGE["prompt_tokens"] = 0
        TOKEN_USAGE["completion_tokens"] = 0
        TOKEN_USAGE["total_tokens"] = 0
        TOKEN_USAGE["by_operation"] = {}


def snapshot_total_token_usage() -> Dict[str, Any]:
    """Copy total token usage accumulated by all worker threads."""
    with TOKEN_USAGE_LOCK:
        return json.loads(json.dumps(TOKEN_USAGE, ensure_ascii=False))


def diff_token_usage(
    start_usage: Dict[str, Any],
    end_usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """计算两个 token usage 快照之间的差值。"""
    end = end_usage or get_thread_token_usage()
    diff: Dict[str, Any] = {
        "calls": int(end.get("calls", 0) or 0) - int(start_usage.get("calls", 0) or 0),
        "prompt_tokens": int(end.get("prompt_tokens", 0) or 0)
        - int(start_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(end.get("completion_tokens", 0) or 0)
        - int(start_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(end.get("total_tokens", 0) or 0)
        - int(start_usage.get("total_tokens", 0) or 0),
        "by_operation": {},
    }

    operation_names = set((start_usage.get("by_operation") or {}).keys())
    operation_names.update((end.get("by_operation") or {}).keys())
    for operation in sorted(operation_names):
        start_op = (start_usage.get("by_operation") or {}).get(operation, {})
        end_op = (end.get("by_operation") or {}).get(operation, {})
        op_diff = {
            "calls": int(end_op.get("calls", 0) or 0)
            - int(start_op.get("calls", 0) or 0),
            "prompt_tokens": int(end_op.get("prompt_tokens", 0) or 0)
            - int(start_op.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(end_op.get("completion_tokens", 0) or 0)
            - int(start_op.get("completion_tokens", 0) or 0),
            "total_tokens": int(end_op.get("total_tokens", 0) or 0)
            - int(start_op.get("total_tokens", 0) or 0),
        }
        if any(value != 0 for value in op_diff.values()):
            diff["by_operation"][operation] = op_diff

    return diff


def add_token_usage(*usage_items: Dict[str, Any]) -> Dict[str, Any]:
    """合并多段 token usage 差值。"""
    total: Dict[str, Any] = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_operation": {},
    }

    for usage in usage_items:
        total["calls"] += int(usage.get("calls", 0) or 0)
        total["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        total["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        total["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

        for operation, op_usage in (usage.get("by_operation") or {}).items():
            if operation not in total["by_operation"]:
                total["by_operation"][operation] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            target = total["by_operation"][operation]
            target["calls"] += int(op_usage.get("calls", 0) or 0)
            target["prompt_tokens"] += int(op_usage.get("prompt_tokens", 0) or 0)
            target["completion_tokens"] += int(op_usage.get("completion_tokens", 0) or 0)
            target["total_tokens"] += int(op_usage.get("total_tokens", 0) or 0)

    return total

