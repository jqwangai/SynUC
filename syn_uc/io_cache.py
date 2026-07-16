"""JSONL and cache helpers for dataset synthesis."""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional


JSONL_WRITE_LOCK = threading.Lock()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。"""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {line_no} 行不是合法 JSON，已跳过：{e}")
    return data


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    """向 JSONL 文件追加一条数据。"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with JSONL_WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_task_filter_cache(path: str) -> Dict[str, Dict[str, Any]]:
    """读取任务分类缓存，按 source_id 映射到最新分类结果。"""
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return cache

    for line_no, record in enumerate(read_jsonl(path), start=1):
        source_id = record.get("source_id")
        filter_result = record.get("filter_result")
        if not source_id or not isinstance(filter_result, dict):
            print(f"[警告] 任务分类缓存第 {line_no} 行格式不完整，已跳过。")
            continue
        cache[str(source_id)] = filter_result

    return cache


def append_task_filter_cache(
    path: str,
    source_id: str,
    model: str,
    filter_result: Dict[str, Any],
    token_usage: Optional[Dict[str, Any]] = None,
) -> None:
    """把一次成功的任务分类结果追加写入缓存。"""
    record = {
        "source_id": source_id,
        "model": model,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filter_result": filter_result,
    }
    if token_usage is not None:
        record["token_usage"] = token_usage
    append_jsonl(path, record)


def load_extract_features_cache(path: str) -> Dict[str, Dict[str, Any]]:
    """读取约束抽取缓存，按 source_id 映射到最新抽取结果。"""
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return cache

    for line_no, record in enumerate(read_jsonl(path), start=1):
        source_id = record.get("source_id")
        features = record.get("features")
        if not source_id or not isinstance(features, dict):
            print(f"[警告] 约束抽取缓存第 {line_no} 行格式不完整，已跳过。")
            continue
        cache[str(source_id)] = features

    return cache


def append_extract_features_cache(
    path: str,
    source_id: str,
    model: str,
    features: Dict[str, Any],
    token_usage: Optional[Dict[str, Any]] = None,
) -> None:
    """把一次成功的约束抽取结果追加写入缓存。"""
    record = {
        "source_id": source_id,
        "model": model,
        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
    }
    if token_usage is not None:
        record["token_usage"] = token_usage
    append_jsonl(path, record)


def normal_extract_features_cache_key(source_id: str) -> str:
    """Return the cache id for normal-dialogue historical constraints."""
    return f"{source_id}_normal"

