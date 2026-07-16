"""Parallel Syn-UC dataset synthesis pipeline.

The implicit-conflict path extracts active historical constraints, retrieves
matching YAML memory patterns with an LLM, generates target constraints,
naturalizes them into follow-up queries, and verifies the results. The explicit
revision path uses compact memory-pattern fields without trajectory examples.

Supported JSONL inputs:
1. {"id": "...", "messages": [{"role": "user", "content": "..."}]}
2. {"conversation_id": "...", "conversation": [{"role": "user", "content": "..."}]}
"""

import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .config import (
    ENABLE_DEEP_THINKING,
    ENABLE_EXPLICIT_REVISION,
    ENABLE_IMPLICIT_CONFLICT,
    ENABLE_NORMAL_DIALOGUE,
    ENABLE_TRAJECTORY_BUFFER,
    ENABLE_TASK_FILTER,
    EXTRACT_FEATURES_CACHE_PATH as EXTRACT_CONSTRAINTS_CACHE_PATH,
    INPUT_PATH,
    MAX_EXPLICIT_PER_DIALOGUE,
    MAX_IMPLICIT_PER_DIALOGUE,
    MEMORY_TEMPLATE_PATH,
    MEMORY_PATH,
    MODEL,
    NORMAL_DIALOGUE_PROBABILITY_FOR_SELECTED_TASK,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OUTPUT_PATH,
    PARALLEL_WORKERS,
    TRAJECTORY_BUFFER_PATH,
    TRAJECTORY_BUFFER_MAX_NEGATIVE_TRAJECTORIES,
    TRAJECTORY_BUFFER_MAX_POSITIVE_TRAJECTORIES,
    TRAJECTORY_BUFFER_POSITIVE_MIN_CONFIDENCE,
    TRAJECTORY_BUFFER_THRESHOLD,
    REASONING_EFFORT,
    SAMPLE_LIMIT,
    SLEEP_SECONDS,
    TASK_FILTER_CACHE_PATH,
    TOKEN_USAGE_PATH,
)
from .dialogue_utils import (
    compact_candidate_id,
    count_user_turns,
    extract_json_from_text,
    messages_to_text,
    normalize_dialogue_item,
    render_prompt,
    safe_json_dumps,
    safe_json_dumps_without_evidence_turn,
    split_for_normal_verification,
)
from .io_cache import (
    append_extract_features_cache as append_extract_constraints_cache,
    append_jsonl,
    append_task_filter_cache,
    load_extract_features_cache as load_extract_constraints_cache,
    load_task_filter_cache,
    normal_extract_features_cache_key as normal_extract_constraints_cache_key,
    read_jsonl,
)
from .memory_ops import (
    load_memory_patterns,
    retrieve_conflict_anchors,
    reset_memory_patterns_cache,
    selected_conflict_anchors_to_text,
)
from .trajectory_buffer import (
    TrajectoryBuffer,
    build_trajectory,
    context_from_constraints,
    trajectory_buffer_event,
    summarize_stats,
)
from .token_usage import (
    TOKEN_USAGE,
    add_token_usage,
    diff_token_usage,
    record_token_usage,
    reset_all_token_usage,
    reset_token_usage,
    snapshot_token_usage,
    snapshot_total_token_usage,
)
from .prompts import (
    EXTRACT_ACTIVE_HISTORICAL_CONSTRAINTS_PROMPT,
    GENERATE_EXPLICIT_REVISION_QUERIES_PROMPT,
    GENERATE_IMPLICIT_NEW_CONSTRAINTS_PROMPT,
    NATURALIZE_IMPLICIT_CONFLICT_QUERIES_PROMPT,
    TASK_FILTER_PROMPT,
    VERIFY_UNITE_PROMPT,
)


# =========================================================
# LLM 调用函数
# =========================================================


def prepare_runtime_memory_file() -> None:
    """Copy the immutable memory template to the runtime area for this run."""
    if MEMORY_TEMPLATE_PATH == MEMORY_PATH:
        raise ValueError("Memory template and runtime paths must be different files.")
    if not MEMORY_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Memory template not found: {MEMORY_TEMPLATE_PATH}")

    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MEMORY_TEMPLATE_PATH, MEMORY_PATH)
    reset_memory_patterns_cache()
    print(f"[Memory] runtime YAML copied: {MEMORY_TEMPLATE_PATH} -> {MEMORY_PATH}")


def build_client() -> OpenAI:
    """构建 OpenAI-compatible 客户端。"""
    if not OPENAI_API_KEY:
        raise ValueError(
            "Missing API key. Set OPENAI_API_KEY or DEEPSEEK_API_KEY before running."
        )
    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# 单步 LLM 操作
# =========================================================


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    operation: str = "unknown",
    temperature: float = 0.2,
    max_tokens: int = 1200,
    retry: int = 3,
    sleep_seconds: float = 2.0,
    json_object_response: bool = True,
) -> Any:
    """调用 LLM，并尽量把输出解析为 JSON。"""
    last_error = None

    for attempt in range(1, retry + 1):
        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_object_response:
                request_kwargs["response_format"] = {"type": "json_object"}
            if ENABLE_DEEP_THINKING:
                request_kwargs["reasoning_effort"] = REASONING_EFFORT
                request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = client.chat.completions.create(**request_kwargs)
            record_token_usage(operation, response)

            content = response.choices[0].message.content
            parsed = extract_json_from_text(content)
            if parsed or isinstance(parsed, list):
                return parsed

            last_error = f"模型输出无法解析为 JSON：{content}"

        except Exception as e:  # noqa: BLE001
            last_error = str(e)

        print(f"[警告] LLM 调用失败，第 {attempt}/{retry} 次：{last_error}")
        time.sleep(sleep_seconds)

    return {"_error": last_error or "unknown error"}



def task_filter(
    client: OpenAI,
    model: str,
    dialogue: List[Dict[str, str]],
) -> Dict[str, Any]:
    """对原始对话进行任务分类。"""
    prompt = render_prompt(
        TASK_FILTER_PROMPT,
        DIALOGUE=messages_to_text(dialogue),
    )
    return call_llm(
        client,
        model,
        prompt,
        operation="task_filter",
        temperature=0.0,
        max_tokens=500,
    )


def should_synthesize_from_filter(filter_result: Dict[str, Any]) -> bool:
    """根据任务分类结果决定是否进入冲突/修订合成流程。"""
    synthesizable_task_types = {
        "writing",
        "coding",
        "planning",
        "information_extraction",
        "summarization",
        "translation",
        "classification_judgement",
    }
    return filter_result.get("task_type") in synthesizable_task_types


def extract_active_constraints(
    client: OpenAI,
    model: str,
    dialogue: List[Dict[str, str]],
) -> Dict[str, Any]:
    """抽取当前任务仍然有效的历史显式/隐式约束。"""
    prompt = render_prompt(
        EXTRACT_ACTIVE_HISTORICAL_CONSTRAINTS_PROMPT,
        DIALOGUE=messages_to_text(dialogue),
    )
    return call_llm(
        client,
        model,
        prompt,
        operation="extract_active_constraints",
        temperature=0.0,
        max_tokens=1800,
    )


def generate_implicit_new_constraints(
    client: OpenAI,
    model: str,
    dialogue: List[Dict[str, str]],
    constraints: Dict[str, Any],
    conflict_anchors: List[Dict[str, Any]],
) -> Any:
    """使用隐式冲突算子生成目标约束 C，并直接具体化为 query。"""
    prompt = render_prompt(
        GENERATE_IMPLICIT_NEW_CONSTRAINTS_PROMPT,
        DIALOGUE=messages_to_text(dialogue),
        HISTORICAL_CONSTRAINTS=safe_json_dumps(constraints),
        CONFLICT_ANCHORS=(
            selected_conflict_anchors_to_text(conflict_anchors)
            if conflict_anchors
            else "[]"
        ),
    )
    return call_llm(
        client,
        model,
        prompt,
        operation="generate_implicit_new_constraints",
        temperature=0.0,
        max_tokens=2000,
        json_object_response=False,
    )


def naturalize_implicit_conflict_queries(
    client: OpenAI,
    model: str,
    dialogue: List[Dict[str, str]],
    constraints: Dict[str, Any],
    constraint_pairs: List[Dict[str, Any]],
    source_language: Any,
) -> Dict[str, Any]:
    """Naturalize implicit-conflict constraint pairs into follow-up queries."""
    prompt = render_prompt(
        NATURALIZE_IMPLICIT_CONFLICT_QUERIES_PROMPT,
        DIALOGUE=messages_to_text(dialogue),
        TARGET_LANGUAGE=str(source_language),
        HISTORICAL_CONSTRAINTS=safe_json_dumps(constraints),
        CONSTRAINT_PAIRS=safe_json_dumps(constraint_pairs),
    )
    return call_llm(
        client,
        model,
        prompt,
        operation="naturalize_implicit_conflict_queries",
        temperature=0.0,
        max_tokens=1800,
    )


def get_active_explicit_constraints_for_memory(
    constraints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return active explicit constraints in the same order Memory will index them."""
    active_constraints = constraints.get("active_constraints")
    if not isinstance(active_constraints, dict):
        return []

    raw_explicit_constraints = active_constraints.get("explicit_constraints")
    if not isinstance(raw_explicit_constraints, list):
        return []

    explicit_constraints: List[Dict[str, Any]] = []
    for item in raw_explicit_constraints:
        if not isinstance(item, dict):
            continue
        constraint = dict(item)
        source = str(constraint.get("source", "explicit") or "explicit").strip().lower()
        if source != "explicit":
            continue
        constraint["source"] = "explicit"
        if str(constraint.get("dimension", "")).strip() and str(
            constraint.get("value", "")
        ).strip():
            explicit_constraints.append(constraint)
    return explicit_constraints


def explicit_constraints_view_for_memory(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Build a constraints object whose Memory-indexed constraints are explicit only."""
    constraints_view = json.loads(json.dumps(constraints, ensure_ascii=False))
    constraints_view["active_constraints"] = {
        "explicit_constraints": get_active_explicit_constraints_for_memory(constraints),
        "implicit_constraints": [],
    }
    return constraints_view


def constraint_identity_key(constraint: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Stable key for matching retrieved anchors back to Memory constraint_id values."""
    return (
        str(constraint.get("source", "")).strip().lower(),
        str(constraint.get("level", "")).strip().lower(),
        str(constraint.get("dimension", "")).strip().lower(),
        str(constraint.get("value", "")).strip().lower(),
    )


def annotate_explicit_revision_anchors(
    anchors: List[Dict[str, Any]],
    constraints_view: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Add explicit-only constraint_id and top-level pattern_id to retrieved anchors."""
    explicit_constraints = get_active_explicit_constraints_for_memory(constraints_view)
    constraint_ids_by_key: Dict[Tuple[str, str, str, str], List[str]] = {}
    for idx, constraint in enumerate(explicit_constraints):
        constraint_ids_by_key.setdefault(constraint_identity_key(constraint), []).append(
            str(idx)
        )

    annotated: List[Dict[str, Any]] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        item = json.loads(json.dumps(anchor, ensure_ascii=False))
        historical_constraint = item.get("historical_constraint") or {}
        if not isinstance(historical_constraint, dict):
            continue
        historical_constraint["source"] = "explicit"
        item["historical_constraint"] = historical_constraint

        key = constraint_identity_key(historical_constraint)
        constraint_ids = constraint_ids_by_key.get(key, [])
        if constraint_ids:
            item["constraint_id"] = constraint_ids.pop(0)

        memory_pattern = item.get("memory_pattern") or item.get("conflict_pattern") or {}
        if isinstance(memory_pattern, dict):
            item["pattern_id"] = str(memory_pattern.get("pattern_id", "")).strip()
        annotated.append(item)
    return annotated


def selected_revision_anchors_to_text(revision_anchors: List[Dict[str, Any]]) -> str:
    """Render compact Memory-selected anchors for explicit-revision generation."""
    prompt_anchors = []
    for idx, anchor in enumerate(revision_anchors):
        historical_constraint = anchor.get("historical_constraint") or {}
        memory_pattern = anchor.get("memory_pattern") or anchor.get("conflict_pattern") or {}
        prompt_anchors.append(
            {
                "anchor_id": str(idx),
                "constraint_id": str(anchor.get("constraint_id", "")),
                "pattern_id": str(
                    anchor.get("pattern_id") or memory_pattern.get("pattern_id", "")
                ),
                "historical_constraint": {
                    "source": "explicit",
                    "level": historical_constraint.get("level", ""),
                    "dimension": historical_constraint.get("dimension", ""),
                    "value": historical_constraint.get("value", ""),
                    "evidence_span": historical_constraint.get("evidence_span", ""),
                    "reason_active": historical_constraint.get("reason_active", ""),
                },
                "revision_pattern": {
                    "definition": memory_pattern.get("definition", ""),
                    "description": memory_pattern.get("description", ""),
                    "historical_constraint_feature": memory_pattern.get(
                        "historical_constraint_feature",
                        memory_pattern.get("historical_constraint", ""),
                    ),
                    "new_constraint_feature": memory_pattern.get(
                        "new_constraint_feature",
                        memory_pattern.get("new_constraint", ""),
                    ),
                    "conflict_focus": memory_pattern.get("conflict_focus", ""),
                },
            }
        )
    return safe_json_dumps(prompt_anchors)


def generate_explicit_revision_queries(
    client: OpenAI,
    model: str,
    dialogue: List[Dict[str, str]],
    constraints: Dict[str, Any],
    revision_anchors: List[Dict[str, Any]],
    source_language: Any,
) -> Dict[str, Any]:
    """Use Memory-selected explicit anchors to generate revision constraints and queries."""
    prompt = render_prompt(
        GENERATE_EXPLICIT_REVISION_QUERIES_PROMPT,
        DIALOGUE=messages_to_text(dialogue),
        TARGET_LANGUAGE=str(source_language),
        HISTORICAL_CONSTRAINTS=safe_json_dumps(constraints),
        REVISION_ANCHORS=selected_revision_anchors_to_text(revision_anchors),
    )
    return call_llm(
        client,
        model,
        prompt,
        operation="generate_explicit_revision_queries",
        temperature=0.0,
        max_tokens=1800,
    )


def verify_unite_queries(
    client: OpenAI,
    model: str,
    expected_label: str,
    dialogue: List[Dict[str, str]],
    constraints: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    operation: str,
    max_tokens: int = 3200,
) -> Dict[str, Any]:
    """使用条件验证 prompt 验证一组共享同一上下文的候选 query。"""
    prompt_candidates = []
    for idx, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id", idx))
        query = (
            candidate.get("subsequent_query")
            or candidate.get("new_query")
            or candidate.get("current_query")
            or ""
        )
        if not query:
            query = candidate.get("query") or ""
            if isinstance(query, dict):
                query = query.get("subsequent_query") or query.get("new_query", "")
        prompt_candidates.append(
            {
                "candidate_id": candidate_id,
                "subsequent_query": query,
                "claimed_historical_constraint": candidate.get(
                    "claimed_historical_constraint",
                    candidate.get("claimed_history_constraint", {}),
                )
                or {},
                "claimed_new_constraint": candidate.get("claimed_new_constraint", {})
                or {},
            }
        )

    prompt = render_prompt(
        VERIFY_UNITE_PROMPT,
        EXPECTED_LABEL=expected_label,
        DIALOGUE=messages_to_text(dialogue),
        HISTORICAL_CONSTRAINTS=safe_json_dumps_without_evidence_turn(constraints),
        CANDIDATES=safe_json_dumps(prompt_candidates),
    )
    return call_llm(
        client,
        model,
        prompt,
        operation=operation,
        temperature=0.0,
        max_tokens=max_tokens,
    )


def verify_implicit_queries(
    client: OpenAI,
    model: str,
    dialogue: List[Dict[str, str]],
    constraints: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    constraint_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """一次性验证隐式冲突候选 query。"""
    claimed_candidates = attach_claimed_constraints_to_candidates(
        candidates,
        constraint_pairs,
    )
    return verify_unite_queries(
        client,
        model,
        "Implicit Conflict",
        dialogue,
        constraints,
        claimed_candidates,
        operation="verify_implicit_queries",
        max_tokens=3200,
    )


def verify_explicit_revision_queries(
    client: OpenAI,
    model: str,
    dialogue: List[Dict[str, str]],
    constraints: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    constraint_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """一次性验证显式修订候选 query。"""
    claimed_candidates = attach_claimed_constraints_to_candidates(
        candidates,
        constraint_pairs,
    )
    return verify_unite_queries(
        client,
        model,
        "Explicit Revision",
        dialogue,
        constraints,
        claimed_candidates,
        operation="verify_explicit_revision_queries",
        max_tokens=3200,
    )


def verify_normal_dialogue(
    client: OpenAI,
    model: str,
    history: List[Dict[str, str]],
    constraints: Dict[str, Any],
    current_query: str,
) -> Dict[str, Any]:
    """验证原始样本最后一轮 user query 是否为 Normal Dialogue。"""
    return verify_unite_queries(
        client,
        model,
        "Normal Dialogue",
        history,
        constraints,
        [{"candidate_id": "0", "subsequent_query": current_query}],
        operation="verify_normal_dialogue",
        max_tokens=2200,
    )


# =========================================================
# 生成结果整理
# =========================================================

def get_legacy_new_constraints(
    generation: Any,
    max_count: int,
) -> List[Dict[str, Any]]:
    """从目标约束生成结果中取出候选，并补齐 candidate_id。"""
    raw_constraints = generation.get("new_constraints")
    if not isinstance(raw_constraints, list):
        raw_constraints = generation.get("candidates")
    if not isinstance(raw_constraints, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_constraints):
        if not isinstance(item, dict):
            continue
        new_constraint = item.get("new_constraint")
        if not isinstance(new_constraint, dict):
            continue

        candidate = dict(item)
        candidate["candidate_id"] = compact_candidate_id(candidate.get("candidate_id"), idx)
        candidate["new_constraint"] = new_constraint
        historical_constraint = (
            candidate.get("historical_constraint")
            or candidate.get("history_constraint")
            or candidate.get("anchor_constraint")
        )
        if isinstance(historical_constraint, dict):
            candidate["historical_constraint"] = historical_constraint
        normalized.append(candidate)
        if len(normalized) >= max_count:
            break

    return normalized


def get_new_constraints(
    generation: Any,
    max_count: int,
) -> List[Dict[str, Any]]:
    """Read implicit-conflict new constraints from a top-level list."""
    raw_constraints = generation
    if isinstance(generation, dict):
        raw_constraints = (
            generation.get("new_constraints")
            or generation.get("introduced_constraints")
            or generation.get("candidates")
        )
    if not isinstance(raw_constraints, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_constraints):
        if not isinstance(item, dict):
            continue
        new_constraint = item.get("new_constraint")
        if not isinstance(new_constraint, dict):
            new_constraint = item.get("introduced_constraint")
        if not isinstance(new_constraint, dict):
            continue

        candidate = dict(item)
        candidate["candidate_id"] = compact_candidate_id(candidate.get("candidate_id"), idx)
        candidate["new_constraint"] = new_constraint
        historical_constraint = (
            candidate.get("historical_constraint")
            or candidate.get("history_constraint")
            or candidate.get("anchor_constraint")
        )
        if isinstance(historical_constraint, dict):
            candidate["historical_constraint"] = historical_constraint
        normalized.append(candidate)
        if len(normalized) >= max_count:
            break

    return normalized


def normalize_query_candidates(
    query_generation: Dict[str, Any],
    max_count: int,
) -> List[Dict[str, Any]]:
    """
    整理 query 生成结果。

    为了兼容下游转换脚本，这里把候选重新编号为 0,1,2...；
    原始 target candidate_id 保存在 source_candidate_id 中。
    """
    raw_queries = query_generation.get("queries")
    if not isinstance(raw_queries, list):
        raw_queries = query_generation.get("candidates")
    if not isinstance(raw_queries, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for idx, query_item in enumerate(raw_queries):
        if not isinstance(query_item, dict):
            continue
        query_value = query_item.get("query", "")
        if isinstance(query_value, dict):
            query_value = query_value.get("new_query", "")
        new_query = str(
            query_item.get("new_query")
            or query_value
            or query_item.get("current_query", "")
        ).strip()
        if not new_query:
            continue

        source_candidate_id = compact_candidate_id(query_item.get("candidate_id"), idx)
        normalized_item = dict(query_item)
        normalized_item["candidate_id"] = str(len(normalized))
        normalized_item["source_candidate_id"] = source_candidate_id
        normalized_item["new_query"] = new_query
        normalized.append(normalized_item)
        if len(normalized) >= max_count:
            break

    return normalized


def query_map_from_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, str]:
    """按 candidate_id 建立 query 映射。"""
    query_map: Dict[str, str] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        new_query = str(candidate.get("new_query", "")).strip()
        if candidate_id and new_query:
            query_map[candidate_id] = new_query
    return query_map


def constraint_map_from_constraints(
    constraints: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """按目标约束 candidate_id 建立映射。"""
    constraint_map: Dict[str, Dict[str, Any]] = {}
    for constraint in constraints:
        candidate_id = str(constraint.get("candidate_id", ""))
        if candidate_id:
            constraint_map[candidate_id] = constraint
    return constraint_map


def minimal_historical_constraint(target: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the claimed historical constraint fields used by verification."""
    historical_constraint = (
        target.get("historical_constraint")
        or target.get("history_constraint")
        or target.get("anchor_constraint")
    )
    if not isinstance(historical_constraint, dict):
        return {}
    return {
        "source": historical_constraint.get("source", ""),
        "level": historical_constraint.get("level", ""),
        "dimension": historical_constraint.get("dimension", ""),
        "value": historical_constraint.get("value", ""),
    }


def minimal_new_constraint(constraint_pair: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the claimed new constraint fields used by verification."""
    new_constraint = (
        constraint_pair.get("new_constraint")
        or constraint_pair.get("introduced_constraint")
    )
    if not isinstance(new_constraint, dict):
        return {}
    return {
        "source": new_constraint.get("source", ""),
        "level": new_constraint.get("level", ""),
        "dimension": new_constraint.get("dimension", ""),
        "value": new_constraint.get("value", ""),
    }


def attach_claimed_constraints_to_candidates(
    candidates: List[Dict[str, Any]],
    constraint_pairs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach claimed history/current constraints to normalized query candidates."""
    constraint_map = constraint_map_from_constraints(constraint_pairs)
    attached: List[Dict[str, Any]] = []
    for candidate in candidates:
        item = json.loads(json.dumps(candidate, ensure_ascii=False))
        source_candidate_id = str(
            item.get("source_candidate_id", item.get("candidate_id", ""))
        )
        constraint_pair = constraint_map.get(source_candidate_id)
        if constraint_pair is None:
            constraint_pair = constraint_map.get(str(item.get("candidate_id", "")), {})
        item["claimed_historical_constraint"] = minimal_historical_constraint(constraint_pair)
        item["claimed_new_constraint"] = minimal_new_constraint(constraint_pair)
        attached.append(item)
    return attached


def get_verification_results(verification_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从统一验证 prompt 的原始 JSON 输出中读取 results。"""
    results = verification_raw.get("results")
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict)]


def find_verification_result(
    verification_raw: Dict[str, Any],
    candidate_id: str = "0",
) -> Dict[str, Any]:
    """按 candidate_id 读取统一验证结果。"""
    for result in get_verification_results(verification_raw):
        if str(result.get("candidate_id", "")) == str(candidate_id):
            return result
    return {}


def is_expected_verification_label(
    verification: Dict[str, Any],
    expected_label: str,
) -> bool:
    """兼容条件验证 prompt 有无 is_valid 字段的判定逻辑。"""
    if verification.get("label") != expected_label:
        return False
    if "is_valid" in verification and not verification.get("is_valid"):
        return False
    return True


def single_result_verification(
    verification_result: Dict[str, Any],
) -> Dict[str, Any]:
    """把批量验证中的单个结果转成 candidate_id=0 的单样本验证结果。"""
    result = json.loads(json.dumps(verification_result, ensure_ascii=False))
    result["candidate_id"] = "0"
    return {"results": [result]}


def attach_verification_to_candidates(
    candidates: List[Dict[str, Any]],
    verification_raw: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """把验证结果按 candidate_id 贴回候选 query，便于保存失败过程。"""
    verification_map = {
        str(result.get("candidate_id", "")): result
        for result in get_verification_results(verification_raw)
    }
    attached: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        item = json.loads(json.dumps(candidate, ensure_ascii=False))
        item["verification"] = verification_map.get(candidate_id, {})
        attached.append(item)
    return attached


def build_implicit_trajectory_buffer_events(
    source_id: str,
    constraints: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    constraint_pairs: List[Dict[str, Any]],
    verification_raw: Dict[str, Any],
    positive_min_confidence: int = TRAJECTORY_BUFFER_POSITIVE_MIN_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """Convert implicit-conflict verification results into trajectory-buffer events."""
    candidate_map = {
        str(candidate.get("candidate_id", "")): candidate
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    constraint_map = constraint_map_from_constraints(constraint_pairs)
    events: List[Dict[str, Any]] = []

    for verification in get_verification_results(verification_raw):
        candidate_id = str(verification.get("candidate_id", ""))
        candidate = candidate_map.get(candidate_id)
        if candidate is None:
            continue

        source_candidate_id = str(
            candidate.get("source_candidate_id", candidate.get("candidate_id", ""))
        )
        constraint_pair = constraint_map.get(source_candidate_id)
        if constraint_pair is None:
            constraint_pair = constraint_map.get(candidate_id, {})

        pattern_id = str(
            constraint_pair.get("pattern_id") or candidate.get("pattern_id") or ""
        ).strip()
        if not pattern_id:
            continue

        historical_constraint = (
            constraint_pair.get("historical_constraint")
            or constraint_pair.get("history_constraint")
            or constraint_pair.get("anchor_constraint")
            or candidate.get("claimed_historical_constraint")
            or {}
        )
        new_constraint = (
            constraint_pair.get("new_constraint")
            or constraint_pair.get("introduced_constraint")
            or candidate.get("claimed_new_constraint")
            or {}
        )
        if not isinstance(historical_constraint, dict) or not isinstance(new_constraint, dict):
            continue

        user_query = str(
            candidate.get("new_query")
            or candidate.get("subsequent_query")
            or candidate.get("current_query")
            or ""
        ).strip()
        if not user_query:
            continue

        is_positive = is_expected_verification_label(verification, "Implicit Conflict")
        if is_positive:
            try:
                confidence = int(verification.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0
            if confidence < positive_min_confidence:
                continue
        trajectory = build_trajectory(
            source_id=source_id,
            candidate_id=candidate_id,
            context=context_from_constraints(constraints, new_constraint),
            historical_constraint=historical_constraint,
            new_constraint=new_constraint,
            user_query=user_query,
            verification=verification,
            is_positive=is_positive,
        )
        events.append(
            trajectory_buffer_event(
                pattern_id=pattern_id,
                trajectory_type="positive" if is_positive else "negative",
                trajectory=trajectory,
            )
        )

    return events


def normalize_generation_for_storage(
    generation: Dict[str, Any],
    normalized_queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """存储生成结果，并用规整后的 queries 兼容后续转换脚本。"""
    stored_generation = json.loads(json.dumps(generation, ensure_ascii=False))
    if isinstance(stored_generation, list):
        stored_generation = {"new_constraints": stored_generation}
    stored_generation["queries"] = normalized_queries
    return stored_generation


def build_raw_implicit_llm_outputs(
    new_constraints_generation: Any,
    query_generation: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep only raw LLM returns for implicit-conflict generation."""
    return {
        "new_constraints_generation": new_constraints_generation,
        "naturalization_generation": query_generation,
    }


def build_raw_explicit_llm_outputs(
    explicit_generation: Dict[str, Any],
    revision_anchors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Keep only raw LLM returns for explicit-revision generation."""
    return {
        "revision_anchors": revision_anchors or [],
        "generation": explicit_generation,
    }


def build_rejected_process_sample(
    source_id: str,
    stage: str,
    source_messages: List[Dict[str, str]],
    llm_outputs: Dict[str, Any],
    token_usage: Dict[str, Any],
    candidates: Optional[List[Dict[str, Any]]] = None,
    current_query: Optional[str] = None,
    note: str = "",
) -> Dict[str, Any]:
    """验证均失败时保存过程信息；该记录不作为训练样本使用。"""
    sample: Dict[str, Any] = {
        "sample_id": f"{source_id}_{stage}_reject",
        "source_id": source_id,
        "label": "Reject",
        "intended_label": {
            "normal": "Normal Dialogue",
            "implicit": "Implicit Conflict",
            "explicit": "Explicit Revision",
        }.get(stage, stage),
        "dialogue_history": source_messages,
        "meta": {
            "is_training_sample": False,
            "rejection_stage": stage,
            "llm_outputs": llm_outputs,
            "token_usage": token_usage,
            "note": note or "条件验证未通过，但保留合成过程信息用于调试和误差分析。",
        },
    }
    if candidates is not None:
        sample["candidate_queries"] = candidates
    if current_query is not None:
        sample["current_query"] = current_query
    return sample


def get_source_language(item: Dict[str, Any], source_id: str) -> Any:
    """Read the source language label from the original INPUT_PATH record."""
    if "language" not in item:
        raise ValueError(f"Missing language in input record: {source_id}")
    return item["language"]


def attach_language_to_samples(
    samples: List[Dict[str, Any]],
    language: Any,
) -> List[Dict[str, Any]]:
    """Ensure every saved sample carries the source language label."""
    for sample in samples:
        sample["language"] = language
    return samples


# =========================================================
# 样本构造函数
# =========================================================


def build_normal_sample(
    source_id: str,
    source_messages: List[Dict[str, str]],
    constraints: Dict[str, Any],
    verification_raw: Dict[str, Any],
    token_usage: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """构造 Normal Dialogue 样本。"""
    verify_result = find_verification_result(verification_raw, candidate_id="0")
    if not is_expected_verification_label(verify_result, "Normal Dialogue"):
        return None

    history, current_query = split_for_normal_verification(source_messages)
    if current_query is None:
        return None

    return {
        "sample_id": f"{source_id}_normal",
        "source_id": source_id,
        "label": "Normal Dialogue",
        "dialogue_history": history,
        "current_query": current_query,
        "meta": {
            "llm_outputs": {
                "extract_constraints": constraints,
                "verify": verification_raw,
            },
            "token_usage": token_usage,
        },
    }


def build_implicit_batch_sample(
    source_id: str,
    source_messages: List[Dict[str, str]],
    constraints: Dict[str, Any],
    implicit_generation: Dict[str, Any],
    normalized_queries: List[Dict[str, Any]],
    batch_verification: Dict[str, Any],
    token_usage: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """把同一原始对话的多条有效隐式冲突候选聚合成一个样本。"""
    query_map = query_map_from_candidates(normalized_queries)
    valid_candidates: List[Dict[str, Any]] = []
    for verification in get_verification_results(batch_verification):
        if not is_expected_verification_label(verification, "Implicit Conflict"):
            continue
        candidate_id = str(verification.get("candidate_id", ""))
        new_query = query_map.get(candidate_id)
        if not new_query:
            continue
        valid_candidates.append(
            {
                "candidate_id": candidate_id,
                "current_query": new_query,
                "verification": verification,
            }
        )

    if not valid_candidates:
        return None

    return {
        "sample_id": f"{source_id}_implicit",
        "source_id": source_id,
        "label": "Implicit Conflict",
        "dialogue_history": source_messages,
        "candidate_queries": valid_candidates,
        "meta": {
            "llm_outputs": {
                "extract_constraints": constraints,
                "generate_implicit": implicit_generation,
                "verify": batch_verification,
            },
            "token_usage": token_usage,
        },
    }


def build_explicit_revision_samples(
    source_id: str,
    source_messages: List[Dict[str, str]],
    constraints: Dict[str, Any],
    query_generation: Dict[str, Any],
    normalized_queries: List[Dict[str, Any]],
    verification_raw: Dict[str, Any],
    token_usage: Dict[str, Any],
    revision_anchors: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """构造 Explicit Revision 样本；每条有效候选单独落一条样本。"""
    query_map = query_map_from_candidates(normalized_queries)
    samples: List[Dict[str, Any]] = []

    for verification in get_verification_results(verification_raw):
        if not is_expected_verification_label(verification, "Explicit Revision"):
            continue

        candidate_id = str(verification.get("candidate_id", ""))
        new_query = query_map.get(candidate_id)
        if not new_query:
            continue

        samples.append(
            {
                "sample_id": f"{source_id}_explicit",
                "source_id": source_id,
                "label": "Explicit Revision",
                "dialogue_history": source_messages,
                "current_query": new_query,
                "meta": {
                    "llm_outputs": {
                        "extract_constraints": constraints,
                        "generate_explicit_revision": build_raw_explicit_llm_outputs(
                            query_generation,
                            revision_anchors,
                        ),
                        "verify": single_result_verification(verification),
                    },
                    "token_usage": token_usage,
                },
            }
        )

    return samples


# =========================================================
# 完整数据合成流程
# =========================================================


def synthesize_normal_from_original(
    client: OpenAI,
    model: str,
    source_id: str,
    messages: List[Dict[str, str]],
    extract_constraints_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    extract_constraints_cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """对原始最后一轮 user query 做 Normal Dialogue 验证。"""
    results: List[Dict[str, Any]] = []
    history, current_query = split_for_normal_verification(messages)
    if current_query is None:
        return results

    normal_token_start = snapshot_token_usage()
    normal_cache_id = normal_extract_constraints_cache_key(source_id)
    if extract_constraints_cache is not None and normal_cache_id in extract_constraints_cache:
        normal_constraints = extract_constraints_cache[normal_cache_id]
        print(f"[cache] {normal_cache_id} reuse normal extract constraints.")
    else:
        normal_constraint_token_start = snapshot_token_usage()
        normal_constraints = extract_active_constraints(client, model, history)
        if normal_constraints and not normal_constraints.get("_error"):
            normal_constraint_token_usage = diff_token_usage(normal_constraint_token_start)
            if extract_constraints_cache is not None:
                extract_constraints_cache[normal_cache_id] = normal_constraints
            if extract_constraints_cache_path:
                append_extract_constraints_cache(
                    extract_constraints_cache_path,
                    normal_cache_id,
                    model,
                    normal_constraints,
                    token_usage=normal_constraint_token_usage,
                )
            print(f"[cache] {normal_cache_id} wrote normal extract constraints.")
    if not normal_constraints or normal_constraints.get("_error"):
        print(f"[跳过] {source_id} Normal Dialogue 约束抽取失败：{normal_constraints}")
        return results

    normal_verify = verify_normal_dialogue(
        client,
        model,
        history,
        normal_constraints,
        current_query,
    )
    normal_sample = build_normal_sample(
        source_id,
        messages,
        normal_constraints,
        normal_verify,
        diff_token_usage(normal_token_start),
    )
    if normal_sample:
        results.append(normal_sample)
    else:
        results.append(
            build_rejected_process_sample(
                source_id=source_id,
                stage="normal",
                source_messages=history,
                current_query=current_query,
                llm_outputs={
                    "extract_constraints": normal_constraints,
                    "verify": normal_verify,
                },
                token_usage=diff_token_usage(normal_token_start),
                note="Normal Dialogue 条件验证未通过，保留原始 query 的验证过程。",
            )
        )

    return results


def synthesize_one_dialogue(
    client: OpenAI,
    model: str,
    item: Dict[str, Any],
    max_implicit_per_dialogue: int = 3,
    max_explicit_per_dialogue: int = 1,
    enable_task_filter: bool = True,
    task_filter_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    task_filter_cache_path: Optional[str] = None,
    extract_constraints_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    extract_constraints_cache_path: Optional[str] = None,
    memory_patterns: Optional[List[Dict[str, Any]]] = None,
    trajectory_buffer_events: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """对单条原始对话合成样本。"""
    source_id, messages = normalize_dialogue_item(item)

    if not messages:
        print(f"[跳过] {source_id} 没有 messages/conversation 字段。")
        return []

    if count_user_turns(messages) <= 1:
        print(f"[跳过] {source_id} 只有 1 轮 user 对话。")
        return []

    source_language = get_source_language(item, source_id)
    results: List[Dict[str, Any]] = []

    # 1. 对任务类型进行分类；非合成任务按需保留 Normal Dialogue 验证路径。
    selected = True
    filter_result: Dict[str, Any] = {"task_type": "not_filtered", "reason": "task filter disabled"}

    if enable_task_filter:
        if task_filter_cache is not None and source_id in task_filter_cache:
            filter_result = task_filter_cache[source_id]
            print(f"[缓存] {source_id} 复用任务分类结果：{filter_result.get('task_type')}")
        else:
            task_filter_token_start = snapshot_token_usage()
            filter_result = task_filter(client, model, messages)
            task_filter_token_usage = diff_token_usage(task_filter_token_start)
            if not filter_result.get("_error"):
                if task_filter_cache is not None:
                    task_filter_cache[source_id] = filter_result
                if task_filter_cache_path:
                    append_task_filter_cache(
                        task_filter_cache_path,
                        source_id,
                        model,
                        filter_result,
                        token_usage=task_filter_token_usage,
                    )
                print(f"[缓存] {source_id} 已写入任务分类缓存：{filter_result.get('task_type')}")
        selected = should_synthesize_from_filter(filter_result)

    if not selected:
        if ENABLE_NORMAL_DIALOGUE:
            print(f"[信息] {source_id} 不进入冲突/修订合成，改为验证 Normal Dialogue：{filter_result.get('reason')}")
            return attach_language_to_samples(
                synthesize_normal_from_original(
                    client,
                    model,
                    source_id,
                    messages,
                    extract_constraints_cache=extract_constraints_cache,
                    extract_constraints_cache_path=extract_constraints_cache_path,
                ),
                source_language,
            )
        print(f"[信息] {source_id} 不进入冲突/修订合成，且 Normal Dialogue 已关闭：{filter_result.get('reason')}")
        return []

    if ENABLE_NORMAL_DIALOGUE and random.random() < NORMAL_DIALOGUE_PROBABILITY_FOR_SELECTED_TASK:
        print(
            f"[信息] {source_id} 命中特定任务，但按 "
            f"{NORMAL_DIALOGUE_PROBABILITY_FOR_SELECTED_TASK:.0%} 概率进入 Normal Dialogue 合成。"
        )
        return attach_language_to_samples(
            synthesize_normal_from_original(
                client,
                model,
                source_id,
                messages,
                extract_constraints_cache=extract_constraints_cache,
                extract_constraints_cache_path=extract_constraints_cache_path,
            ),
            source_language,
        )

    # 2. 抽取完整原始对话的历史有效约束。
    constraint_token_start = snapshot_token_usage()
    if extract_constraints_cache is not None and source_id in extract_constraints_cache:
        constraints = extract_constraints_cache[source_id]
        print(f"[缓存] {source_id} 复用历史有效约束抽取结果。")
    else:
        constraints = extract_active_constraints(client, model, messages)
        if constraints and not constraints.get("_error"):
            if extract_constraints_cache is not None:
                extract_constraints_cache[source_id] = constraints
            if extract_constraints_cache_path:
                append_extract_constraints_cache(
                    extract_constraints_cache_path,
                    source_id,
                    model,
                    constraints,
                    token_usage=diff_token_usage(constraint_token_start),
                )
            print(f"[缓存] {source_id} 已写入历史有效约束抽取缓存。")

    if not constraints or constraints.get("_error"):
        print(f"[跳过] {source_id} 历史有效约束抽取失败：{constraints}")
        return results
    constraint_token_usage = diff_token_usage(constraint_token_start)

    # 3. 隐式冲突：在一个 prompt 中完成 H 锚定、O(H, τ, γ) 生成 C、C 具体化为 query。
    if ENABLE_IMPLICIT_CONFLICT and max_implicit_per_dialogue > 0:
        implicit_token_start = snapshot_token_usage()
        conflict_anchors = retrieve_conflict_anchors(
            client,
            model,
            constraints,
            dialogue_text=messages_to_text(messages),
            memory_patterns=memory_patterns,
        )
        if not conflict_anchors:
            print(f"[Memory] {source_id} no conflict anchors for implicit conflict.")
        new_constraints_generation = generate_implicit_new_constraints(
            client,
            model,
            messages,
            constraints,
            conflict_anchors,
        )
        constraint_pairs = get_new_constraints(
            new_constraints_generation,
            max_count=max_implicit_per_dialogue,
        )
        query_generation: Dict[str, Any] = {}
        implicit_queries: List[Dict[str, Any]] = []
        if constraint_pairs:
            query_generation = naturalize_implicit_conflict_queries(
                client,
                model,
                messages,
                constraints,
                constraint_pairs,
                source_language,
            )
            implicit_queries = normalize_query_candidates(
                query_generation,
                max_count=max_implicit_per_dialogue,
            )

        if constraint_pairs and implicit_queries:
            claimed_implicit_queries = attach_claimed_constraints_to_candidates(
                implicit_queries,
                constraint_pairs,
            )
            implicit_verification = verify_implicit_queries(
                client,
                model,
                messages,
                constraints,
                claimed_implicit_queries,
                constraint_pairs,
            )
            if trajectory_buffer_events is not None:
                trajectory_buffer_events.extend(
                    build_implicit_trajectory_buffer_events(
                        source_id=source_id,
                        constraints=constraints,
                        candidates=claimed_implicit_queries,
                        constraint_pairs=constraint_pairs,
                        verification_raw=implicit_verification,
                    )
                )
            implicit_generation_for_storage = build_raw_implicit_llm_outputs(
                new_constraints_generation,
                query_generation,
            )
            implicit_token_usage = add_token_usage(
                constraint_token_usage,
                diff_token_usage(implicit_token_start),
            )
            implicit_sample = build_implicit_batch_sample(
                source_id=source_id,
                source_messages=messages,
                constraints=constraints,
                implicit_generation=implicit_generation_for_storage,
                normalized_queries=claimed_implicit_queries,
                batch_verification=implicit_verification,
                token_usage=implicit_token_usage,
            )
            if implicit_sample:
                results.append(implicit_sample)
            else:
                results.append(
                    build_rejected_process_sample(
                        source_id=source_id,
                        stage="implicit",
                        source_messages=messages,
                        candidates=attach_verification_to_candidates(
                            claimed_implicit_queries,
                            implicit_verification,
                        ),
                        llm_outputs={
                            "extract_constraints": constraints,
                            "generate_implicit": implicit_generation_for_storage,
                            "verify": implicit_verification,
                        },
                        token_usage=implicit_token_usage,
                        note="Implicit Conflict 候选均未通过条件验证，保留生成与验证过程。",
                    )
                )
        elif not constraint_pairs:
            print(f"[信息] {source_id} 未生成可用的隐式目标约束。")
        else:
            print(f"[信息] {source_id} 隐式目标约束未能具体化为 query。")

    # 4. 显式修订：先用 Memory 检索显式 H_E 锚点，再直接替换 C_E 并具体化为 query。
    if ENABLE_EXPLICIT_REVISION and max_explicit_per_dialogue > 0:
        explicit_token_start = snapshot_token_usage()
        explicit_constraints_for_memory = explicit_constraints_view_for_memory(
            constraints
        )
        explicit_anchors = retrieve_conflict_anchors(
            client,
            model,
            explicit_constraints_for_memory,
            dialogue_text=messages_to_text(messages),
            memory_patterns=memory_patterns,
        )
        explicit_anchors = annotate_explicit_revision_anchors(
            explicit_anchors,
            explicit_constraints_for_memory,
        )
        if not explicit_anchors:
            print(f"[Memory] {source_id} no explicit anchors for explicit revision.")

        explicit_generation: Dict[str, Any] = {}
        if explicit_anchors:
            explicit_generation = generate_explicit_revision_queries(
                client,
                model,
                messages,
                constraints,
                explicit_anchors,
                source_language,
            )
        explicit_new_constraints = get_new_constraints(
            explicit_generation,
            max_count=max_explicit_per_dialogue,
        )
        explicit_queries = normalize_query_candidates(
            explicit_generation,
            max_count=max_explicit_per_dialogue,
        )

        if explicit_new_constraints and explicit_queries:
            claimed_explicit_queries = attach_claimed_constraints_to_candidates(
                explicit_queries,
                explicit_new_constraints,
            )
            explicit_verification = verify_explicit_revision_queries(
                client,
                model,
                messages,
                constraints,
                claimed_explicit_queries,
                explicit_new_constraints,
            )
            explicit_token_usage = add_token_usage(
                constraint_token_usage,
                diff_token_usage(explicit_token_start),
            )
            explicit_samples = build_explicit_revision_samples(
                source_id=source_id,
                source_messages=messages,
                constraints=constraints,
                query_generation=explicit_generation,
                normalized_queries=claimed_explicit_queries,
                verification_raw=explicit_verification,
                token_usage=explicit_token_usage,
                revision_anchors=explicit_anchors,
            )
            if explicit_samples:
                results.extend(explicit_samples)
            else:
                results.append(
                    build_rejected_process_sample(
                        source_id=source_id,
                        stage="explicit",
                        source_messages=messages,
                        candidates=attach_verification_to_candidates(
                            claimed_explicit_queries,
                            explicit_verification,
                        ),
                        llm_outputs={
                            "extract_constraints": constraints,
                            "generate_explicit_revision": build_raw_explicit_llm_outputs(
                                explicit_generation,
                                explicit_anchors,
                            ),
                            "verify": explicit_verification,
                        },
                        token_usage=explicit_token_usage,
                        note="Explicit Revision 候选均未通过条件验证，保留生成与验证过程。",
                    )
                )
        elif not explicit_anchors:
            pass
        elif not explicit_new_constraints:
            print(f"[信息] {source_id} 未生成可用的显式修订目标约束。")
        else:
            print(f"[信息] {source_id} 显式目标约束未能具体化为 query。")

    return attach_language_to_samples(results, source_language)


def iter_with_progress(items: List[Dict[str, Any]]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """按需包一层 tqdm。"""
    if tqdm is None:
        return enumerate(items, start=1)
    return enumerate(tqdm(items, desc="合成进度", unit="样本"), start=1)


def synthesize_dataset() -> None:
    """处理配置区指定的整个数据集，串行执行。"""
    prepare_runtime_memory_file()
    reset_token_usage()
    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    client = build_client()
    memory_patterns: Optional[List[Dict[str, Any]]] = None
    if (
        (ENABLE_IMPLICIT_CONFLICT and MAX_IMPLICIT_PER_DIALOGUE > 0)
        or (ENABLE_EXPLICIT_REVISION and MAX_EXPLICIT_PER_DIALOGUE > 0)
    ):
        memory_patterns = load_memory_patterns()
        print(f"[Memory] loaded YAML memory patterns={len(memory_patterns)}")
    trajectory_buffer: Optional[TrajectoryBuffer] = None
    trajectory_buffer_stats = {"added": 0, "promoted": 0, "skipped": 0}
    if ENABLE_TRAJECTORY_BUFFER and ENABLE_IMPLICIT_CONFLICT:
        trajectory_buffer = TrajectoryBuffer(
            TRAJECTORY_BUFFER_PATH,
            MEMORY_PATH,
            threshold=TRAJECTORY_BUFFER_THRESHOLD,
            llm_client=client,
            model=MODEL,
            max_positive_trajectories=TRAJECTORY_BUFFER_MAX_POSITIVE_TRAJECTORIES,
            max_negative_trajectories=TRAJECTORY_BUFFER_MAX_NEGATIVE_TRAJECTORIES,
        )
        print(f"[TrajectoryBuffer] enabled: {TRAJECTORY_BUFFER_PATH}")
    task_filter_cache = load_task_filter_cache(TASK_FILTER_CACHE_PATH)
    extract_constraints_cache = load_extract_constraints_cache(EXTRACT_CONSTRAINTS_CACHE_PATH)

    if ENABLE_TASK_FILTER:
        print(f"[信息] 已加载任务分类缓存 {len(task_filter_cache)} 条：{TASK_FILTER_CACHE_PATH}")
    print(f"[信息] 已加载历史有效约束抽取缓存 {len(extract_constraints_cache)} 条：{EXTRACT_CONSTRAINTS_CACHE_PATH}")

    raw_data = read_jsonl(INPUT_PATH)
    original_count = len(raw_data)
    eligible_data = [
        item
        for item in raw_data
        if count_user_turns(normalize_dialogue_item(item)[1]) > 1
    ]
    filtered_count = original_count - len(eligible_data)
    if SAMPLE_LIMIT is not None:
        eligible_data = eligible_data[:SAMPLE_LIMIT]

    if filtered_count > 0:
        print(f"[信息] 数据提取阶段已过滤 {filtered_count} 条只有 1 轮 user 对话的样本。")

    if os.path.exists(OUTPUT_PATH):
        print(f"[警告] 输出文件 {OUTPUT_PATH} 已存在，将追加数据。")

    print(f"[信息] 待处理样本数：{len(eligible_data)}")

    total_written = 0
    for i, item in iter_with_progress(eligible_data):
        source_id, _ = normalize_dialogue_item(item)
        print(f"\n========== 处理第 {i}/{len(eligible_data)} 条 | {source_id} ==========")
        trajectory_events: List[Dict[str, Any]] = []
        samples = synthesize_one_dialogue(
            client=client,
            model=MODEL,
            item=item,
            max_implicit_per_dialogue=MAX_IMPLICIT_PER_DIALOGUE,
            max_explicit_per_dialogue=MAX_EXPLICIT_PER_DIALOGUE,
            enable_task_filter=ENABLE_TASK_FILTER,
            task_filter_cache=task_filter_cache,
            task_filter_cache_path=TASK_FILTER_CACHE_PATH,
            extract_constraints_cache=extract_constraints_cache,
            extract_constraints_cache_path=EXTRACT_CONSTRAINTS_CACHE_PATH,
            memory_patterns=memory_patterns,
            trajectory_buffer_events=trajectory_events,
        )
        if trajectory_buffer is not None and trajectory_events:
            stats = trajectory_buffer.add_events(trajectory_events)
            for key, value in stats.items():
                trajectory_buffer_stats[key] = trajectory_buffer_stats.get(key, 0) + value
            print(f"[TrajectoryBuffer] {source_id} | {summarize_stats(stats)}")

        for sample in samples:
            append_jsonl(OUTPUT_PATH, sample)
            total_written += 1
            print(f"[写入] {sample['sample_id']} | {sample['label']}")

        if not samples:
            print(f"[信息] 第 {i} 条 {source_id} 未产出样本。")

        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)

    usage_record = {
        "run_started_at": run_started_at,
        "run_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "input_path": INPUT_PATH,
        "output_path": OUTPUT_PATH,
        "task_filter_cache_path": TASK_FILTER_CACHE_PATH,
        "extract_constraints_cache_path": EXTRACT_CONSTRAINTS_CACHE_PATH,
        "input_count": original_count,
        "filtered_single_turn_count": filtered_count,
        "processed_count": len(eligible_data),
        "written_count": total_written,
        "sample_limit": SAMPLE_LIMIT,
        "enable_implicit_conflict": ENABLE_IMPLICIT_CONFLICT,
        "enable_explicit_revision": ENABLE_EXPLICIT_REVISION,
        "enable_normal_dialogue": ENABLE_NORMAL_DIALOGUE,
        "max_implicit_per_dialogue": MAX_IMPLICIT_PER_DIALOGUE,
        "max_explicit_per_dialogue": MAX_EXPLICIT_PER_DIALOGUE,
        "trajectory_buffer": trajectory_buffer_stats,
        "token_usage": TOKEN_USAGE,
    }
    append_jsonl(TOKEN_USAGE_PATH, usage_record)

    print(f"\n完成。共写入 {total_written} 条样本到：{OUTPUT_PATH}")
    print(
        "[Token] "
        f"calls={TOKEN_USAGE['calls']} | "
        f"prompt={TOKEN_USAGE['prompt_tokens']} | "
        f"completion={TOKEN_USAGE['completion_tokens']} | "
        f"total={TOKEN_USAGE['total_tokens']}"
    )
    print(f"[Token] 统计已追加写入：{TOKEN_USAGE_PATH}")
    if trajectory_buffer is not None:
        print(f"[TrajectoryBuffer] run stats | {summarize_stats(trajectory_buffer_stats)}")

def iter_completed_with_progress(futures: Iterable[Any], total: int) -> Iterable[Any]:
    """Wrap completed futures with tqdm when available."""
    if tqdm is None:
        return futures
    return tqdm(futures, total=total, desc="并行合成进度", unit="样本")


def synthesize_dialogue_worker(
    index: int,
    total: int,
    item: Dict[str, Any],
    task_filter_cache: Dict[str, Dict[str, Any]],
    extract_constraints_cache: Dict[str, Dict[str, Any]],
    memory_patterns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Process one dialogue in one worker thread."""
    reset_token_usage()
    source_id, _ = normalize_dialogue_item(item)
    started_at = time.perf_counter()
    client = build_client()
    trajectory_events: List[Dict[str, Any]] = []

    print(f"\n========== 并行处理第 {index}/{total} 条 | {source_id} ==========")
    try:
        samples = synthesize_one_dialogue(
            client=client,
            model=MODEL,
            item=item,
            max_implicit_per_dialogue=MAX_IMPLICIT_PER_DIALOGUE,
            max_explicit_per_dialogue=MAX_EXPLICIT_PER_DIALOGUE,
            enable_task_filter=ENABLE_TASK_FILTER,
            task_filter_cache=task_filter_cache,
            task_filter_cache_path=TASK_FILTER_CACHE_PATH,
            extract_constraints_cache=extract_constraints_cache,
            extract_constraints_cache_path=EXTRACT_CONSTRAINTS_CACHE_PATH,
            memory_patterns=memory_patterns,
            trajectory_buffer_events=trajectory_events,
        )
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
        return {
            "ok": True,
            "index": index,
            "source_id": source_id,
            "samples": samples,
            "trajectory_buffer_events": trajectory_events,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "index": index,
            "source_id": source_id,
            "samples": [],
            "trajectory_buffer_events": trajectory_events,
            "elapsed_seconds": time.perf_counter() - started_at,
            "error": repr(exc),
        }


def synthesize_dataset_parallel() -> None:
    """按样本级并行处理配置区指定的数据集。"""
    if PARALLEL_WORKERS < 1:
        raise ValueError("PARALLEL_WORKERS must be >= 1")

    prepare_runtime_memory_file()
    reset_all_token_usage()
    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    memory_patterns: Optional[List[Dict[str, Any]]] = None
    if (
        (ENABLE_IMPLICIT_CONFLICT and MAX_IMPLICIT_PER_DIALOGUE > 0)
        or (ENABLE_EXPLICIT_REVISION and MAX_EXPLICIT_PER_DIALOGUE > 0)
    ):
        memory_patterns = load_memory_patterns()
        print(f"[Memory] loaded YAML memory patterns={len(memory_patterns)}")
    trajectory_buffer: Optional[TrajectoryBuffer] = None
    trajectory_buffer_stats = {"added": 0, "promoted": 0, "skipped": 0}
    if ENABLE_TRAJECTORY_BUFFER and ENABLE_IMPLICIT_CONFLICT:
        trajectory_buffer_client = build_client()
        trajectory_buffer = TrajectoryBuffer(
            TRAJECTORY_BUFFER_PATH,
            MEMORY_PATH,
            threshold=TRAJECTORY_BUFFER_THRESHOLD,
            llm_client=trajectory_buffer_client,
            model=MODEL,
            max_positive_trajectories=TRAJECTORY_BUFFER_MAX_POSITIVE_TRAJECTORIES,
            max_negative_trajectories=TRAJECTORY_BUFFER_MAX_NEGATIVE_TRAJECTORIES,
        )
        print(f"[TrajectoryBuffer] enabled: {TRAJECTORY_BUFFER_PATH}")
    task_filter_cache = load_task_filter_cache(TASK_FILTER_CACHE_PATH)
    extract_constraints_cache = load_extract_constraints_cache(EXTRACT_CONSTRAINTS_CACHE_PATH)

    if ENABLE_TASK_FILTER:
        print(
            f"[信息] 已加载任务分类缓存 {len(task_filter_cache)} 条："
            f"{TASK_FILTER_CACHE_PATH}"
        )
    print(
        f"[信息] 已加载历史有效约束抽取缓存 {len(extract_constraints_cache)} 条："
        f"{EXTRACT_CONSTRAINTS_CACHE_PATH}"
    )

    raw_data = read_jsonl(INPUT_PATH)
    original_count = len(raw_data)
    eligible_data = [
        item
        for item in raw_data
        if count_user_turns(normalize_dialogue_item(item)[1]) > 1
    ]
    filtered_count = original_count - len(eligible_data)
    if SAMPLE_LIMIT is not None:
        eligible_data = eligible_data[:SAMPLE_LIMIT]

    if filtered_count > 0:
        print(f"[信息] 数据提取阶段已过滤 {filtered_count} 条只有 1 轮 user 对话的样本。")

    if os.path.exists(OUTPUT_PATH):
        print(f"[警告] 输出文件 {OUTPUT_PATH} 已存在，将追加数据。")

    total = len(eligible_data)
    print(f"[信息] 待处理样本数：{total}")
    print(f"[信息] 样本级并行数：{PARALLEL_WORKERS}")

    total_written = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = [
            executor.submit(
                synthesize_dialogue_worker,
                index,
                total,
                item,
                task_filter_cache,
                extract_constraints_cache,
                memory_patterns,
            )
            for index, item in enumerate(eligible_data, start=1)
        ]

        completed = as_completed(futures)
        for future in iter_completed_with_progress(completed, total):
            result = future.result()
            source_id = result["source_id"]
            if not result["ok"]:
                failed_count += 1
                print(f"[错误] {source_id} 处理失败：{result['error']}")
                continue

            trajectory_events = result.get("trajectory_buffer_events", [])
            if trajectory_buffer is not None and trajectory_events:
                stats = trajectory_buffer.add_events(trajectory_events)
                for key, value in stats.items():
                    trajectory_buffer_stats[key] = trajectory_buffer_stats.get(key, 0) + value
                print(f"[TrajectoryBuffer] {source_id} | {summarize_stats(stats)}")

            samples = result["samples"]
            for sample in samples:
                append_jsonl(OUTPUT_PATH, sample)
                total_written += 1
                print(f"[写入] {sample['sample_id']} | {sample['label']}")

            if not samples:
                print(f"[信息] {source_id} 未产出样本。")
            print(
                f"[完成] {source_id} | samples={len(samples)} | "
                f"elapsed={result['elapsed_seconds']:.1f}s"
            )

    total_token_usage = snapshot_total_token_usage()
    usage_record = {
        "run_started_at": run_started_at,
        "run_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "input_path": INPUT_PATH,
        "output_path": OUTPUT_PATH,
        "task_filter_cache_path": TASK_FILTER_CACHE_PATH,
        "extract_constraints_cache_path": EXTRACT_CONSTRAINTS_CACHE_PATH,
        "input_count": original_count,
        "filtered_single_turn_count": filtered_count,
        "processed_count": total,
        "written_count": total_written,
        "failed_count": failed_count,
        "sample_limit": SAMPLE_LIMIT,
        "parallel_workers": PARALLEL_WORKERS,
        "enable_implicit_conflict": ENABLE_IMPLICIT_CONFLICT,
        "enable_explicit_revision": ENABLE_EXPLICIT_REVISION,
        "enable_normal_dialogue": ENABLE_NORMAL_DIALOGUE,
        "max_implicit_per_dialogue": MAX_IMPLICIT_PER_DIALOGUE,
        "max_explicit_per_dialogue": MAX_EXPLICIT_PER_DIALOGUE,
        "trajectory_buffer": trajectory_buffer_stats,
        "token_usage": total_token_usage,
    }
    append_jsonl(TOKEN_USAGE_PATH, usage_record)

    print(f"\n完成。共写入 {total_written} 条样本到：{OUTPUT_PATH}")
    print(
        "[Token] "
        f"calls={total_token_usage['calls']} | "
        f"prompt={total_token_usage['prompt_tokens']} | "
        f"completion={total_token_usage['completion_tokens']} | "
        f"total={total_token_usage['total_tokens']}"
    )
    if failed_count:
        print(f"[警告] 失败样本数：{failed_count}")
    print(f"[Token] 统计已追加写入：{TOKEN_USAGE_PATH}")
    if trajectory_buffer is not None:
        print(f"[TrajectoryBuffer] run stats | {summarize_stats(trajectory_buffer_stats)}")


def main() -> None:
    """本地运行入口。"""
    synthesize_dataset_parallel()


if __name__ == "__main__":
    main()
