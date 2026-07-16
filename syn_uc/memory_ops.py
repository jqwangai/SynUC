"""Implicit-conflict operator memory loading and LLM-based retrieval."""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
import yaml

from .config import (
    ENABLE_DEEP_THINKING,
    MEMORY_DEDUP_BY_DIMENSION,
    MEMORY_PATH,
    MEMORY_RETRIEVAL_MIN_SCORE,
    MEMORY_RETRIEVAL_TOP_K_PER_CONSTRAINT,
    MEMORY_SELECTED_PAIR_LIMIT,
    REASONING_EFFORT,
)
from .token_usage import record_token_usage


MEMORY_LOAD_LOCK = threading.Lock()
MEMORY_PATTERNS_CACHE: Optional[List[Dict[str, Any]]] = None


LLM_MEMORY_RETRIEVAL_PROMPT = """
You are an implicit-conflict pattern retriever. Using the historical dialogue context and the extracted active historical constraints, select the anchors from the candidate memory patterns that are most suitable for constructing an implicit-conflict query.

## Implicit-Conflict Definition
Within the same task or a naturally continued task, a conflict arises when new_query does not explicitly state that earlier requirements should be modified, but the new_constraint it introduces is incompatible with a still-active constraint in historical_constraint.
It counts as a conflict only when the two constraints cannot be satisfied at the same time. "Different" does not mean "conflicting"; "more detailed" is not necessarily conflicting; "newly added content" is also not necessarily conflicting.

Possible paths:
1. H_I_N_E: historical implicit constraint vs new explicit constraint introduced by new_query.
2. H_E_N_I: historical explicit constraint vs new implicit constraint introduced by new_query.
3. H_I_N_I: historical implicit constraint vs new implicit constraint introduced by new_query.
Prioritize generating H_E_N_I and H_I_N_I. Use H_I_N_E only when new_query must explicitly state a specific requirement and that requirement conflicts with a historical implicit constraint.

## Constraint Definitions
- explicit constraints: Requirements that the user directly expresses in new_query and that can be clearly identified from the literal text. They include, but are not limited to, topic, object scope, language, format, length, style, audience, output form, technology choice, input data, and specific content restrictions.
- implicit constraints: Default requirements that the user does not directly state in new_query, but that can be naturally and stably inferred from the task goal, task type, use case, entity attributes, domain norms, or common sense. Implicit constraints are used to supplement task understanding, but requirements that the user has already stated explicitly must not be redundantly recorded as implicit constraints.

## Retrieval Objective
1. historical_constraint refers to constraints from the historical dialogue that are still active and that new_query must continue to follow.
2. memory_pattern represents a reusable construction pattern of "historical constraint feature -> incompatibility with new_query new constraint feature".
3. You need to determine which memory_pattern can naturally start from a given historical_constraint, introduce a new_constraint that may be implied by new_query, and make that new_constraint impossible to satisfy together with historical_constraint, thereby meeting the implicit-conflict definition.
4. The historical dialogue should be used only as background for understanding the task context, entity references, and constraint persistence. The final selection must use constraints from the "active historical constraints" list as anchors; do not independently recover constraints from the historical dialogue that have expired or were not extracted.

## Selection Criteria
1. Prefer items where the historical constraint dimension/value highly matches the pattern's historical_constraint_feature and retrieval_terms.
2. Select only items that can form a truly incompatible relationship. Items that are merely more detailed, richer, slightly broader in scope, slightly different in tone, or that add ordinary content should not be selected.
3. Prefer constraints at the current_subtask level, followed by constraints at the current_main_task level.
4. Output only <<SELECTED_PAIR_LIMIT>> anchors, sorted by suitability from high to low.

All text output must be in English.

Output only valid JSON and no explanatory text. score must be a number from 0 to 1, indicating how suitable this pattern is for constructing an implicit conflict with the historical constraint.

Output format:
{
  "anchors": [
    {
      "constraint_id": "0",
      "pattern_id": "pattern_id",
      "score": 0.86,
      "reason": "Briefly explain why this pattern fits this historical constraint"
    }
  ]
}

Candidate memory patterns:
<<MEMORY_PATTERNS>>

Historical dialogue:
<<DIALOGUE>>

Active historical constraints:
<<HISTORICAL_CONSTRAINTS>>
""".strip()


def reset_memory_patterns_cache() -> None:
    """Clear the default memory-pattern cache after replacing the runtime YAML."""
    global MEMORY_PATTERNS_CACHE

    with MEMORY_LOAD_LOCK:
        MEMORY_PATTERNS_CACHE = None


def render_prompt(template: str, **kwargs: str) -> str:
    prompt = template
    for key, value in kwargs.items():
        prompt = prompt.replace(f"<<{key}>>", value)
    return prompt


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def extract_json_from_text(text: Optional[str]) -> Any:
    if text is None:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def call_llm_json(
    client: OpenAI,
    model: str,
    prompt: str,
    operation: str,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    retry: int = 3,
    sleep_seconds: float = 2.0,
    json_object_response: bool = True,
) -> Any:
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
            parsed = extract_json_from_text(response.choices[0].message.content)
            if parsed:
                return parsed
            last_error = "model output is not valid JSON"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        print(f"[Warning] LLM memory retrieval failed {attempt}/{retry}: {last_error}")
        time.sleep(sleep_seconds)

    return {"_error": last_error or "unknown error"}


def load_memory_patterns(path: Path = MEMORY_PATH) -> List[Dict[str, Any]]:
    """Load and flatten implicit-conflict memory patterns."""
    global MEMORY_PATTERNS_CACHE

    if path == MEMORY_PATH and MEMORY_PATTERNS_CACHE is not None:
        return MEMORY_PATTERNS_CACHE

    with MEMORY_LOAD_LOCK:
        if path == MEMORY_PATH and MEMORY_PATTERNS_CACHE is not None:
            return MEMORY_PATTERNS_CACHE

        with path.open("r", encoding="utf-8") as f:
            memory = yaml.safe_load(f)
        if not isinstance(memory, dict):
            raise ValueError(f"Memory file must be a YAML mapping: {path}")

        operators = memory.get("operators")
        if not isinstance(operators, list):
            raise ValueError(f"Memory file missing list field 'operators': {path}")

        patterns: List[Dict[str, Any]] = []
        for op_idx, operator in enumerate(operators):
            if not isinstance(operator, dict):
                raise ValueError(f"Operator at index {op_idx} must be a mapping: {path}")
            operator_id = str(operator.get("operator_id", "")).strip()
            definition = str(operator.get("definition", "")).strip()
            raw_patterns = operator.get("patterns", [])
            if not isinstance(raw_patterns, list):
                continue
            for pat_idx, pattern in enumerate(raw_patterns):
                if not isinstance(pattern, dict):
                    raise ValueError(
                        f"Pattern at operator {operator_id} index {pat_idx} must be a mapping: {path}"
                    )
                pattern_id = str(pattern.get("pattern_id", "")).strip()
                if not operator_id or not pattern_id:
                    continue
                item = {
                    "memory_key": f"{operator_id}/{pattern_id}",
                    "operator_id": operator_id,
                    "definition": definition,
                    "pattern_id": pattern_id,
                    "pattern": pattern,
                }
                item["retrieval_text"] = pattern_retrieval_text(item)
                patterns.append(item)

        if path == MEMORY_PATH:
            MEMORY_PATTERNS_CACHE = patterns
        return patterns


def pattern_retrieval_text(memory_item: Dict[str, Any]) -> str:
    pattern = memory_item.get("pattern") or {}
    return str(pattern.get("retrieval_terms", "")).strip()


def pattern_historical_constraint_feature(pattern: Dict[str, Any]) -> Any:
    """Return the abstract historical-side feature for a memory pattern."""
    return pattern.get(
        "historical_constraint_feature",
        pattern.get("historical_constraint", ""),
    )


def pattern_new_constraint_feature(pattern: Dict[str, Any]) -> Any:
    """Return the abstract new-query-side feature for a memory pattern."""
    return pattern.get(
        "new_constraint_feature",
        pattern.get("new_constraint", ""),
    )


def get_active_constraints(constraints_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return active explicit and implicit historical constraints."""
    active_constraints = constraints_doc.get("active_constraints") or {}
    constraints: List[Dict[str, Any]] = []
    for key in ("explicit_constraints", "implicit_constraints"):
        items = active_constraints.get(key) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                constraint = dict(item)
                if "source" not in constraint:
                    constraint["source"] = (
                        "explicit" if key == "explicit_constraints" else "implicit"
                    )
                constraints.append(constraint)
    return constraints


def minimal_constraint_for_memory(constraint: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields needed by retrieval and generation."""
    return {
        "source": constraint.get("source", ""),
        "level": constraint.get("level", ""),
        "dimension": constraint.get("dimension", ""),
        "value": constraint.get("value", ""),
        "evidence_span": constraint.get("evidence_span", ""),
        "reason_active": constraint.get("reason_active", ""),
    }


def minimal_pattern_for_memory(memory_item: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields needed by the generator for a selected memory pattern."""
    pattern = memory_item.get("pattern") or {}
    return {
        "operator_id": memory_item.get("operator_id", ""),
        "pattern_id": memory_item.get("pattern_id", ""),
        "definition": memory_item.get("definition", ""),
        "description": pattern.get("description", ""),
        "historical_constraint_feature": pattern_historical_constraint_feature(pattern),
        "new_constraint_feature": pattern_new_constraint_feature(pattern),
        "conflict_focus": pattern.get("conflict_focus", ""),
        "positive_trajectories": pattern.get("positive_trajectories", []),
        "negative_trajectories": pattern.get("negative_trajectories", []),
    }


def pattern_for_retrieval_prompt(memory_item: Dict[str, Any]) -> Dict[str, Any]:
    """Compact one memory pattern for the LLM selector prompt."""
    pattern = memory_item.get("pattern") or {}
    return compact_prompt_mapping(
        {
            "pattern_id": memory_item.get("pattern_id", ""),
            "description": pattern.get("description", ""),
            "historical_constraint_feature": pattern_historical_constraint_feature(pattern),
            "retrieval_terms": pattern.get("retrieval_terms", ""),
        }
    )


def normalize_dedup_key(value: Any) -> str:
    return str(value or "").strip().lower()


def constraint_level_priority(level: Any) -> int:
    normalized = normalize_dedup_key(level).replace("-", "_")
    priorities = {
        "current_sub_task": 2,
        "current_subtask": 2,
        "current_main_task": 1,
        "current_maintask": 1,
    }
    return priorities.get(normalized, 0)


def conflict_anchor_sort_key(anchor: Dict[str, Any]) -> Tuple[int, float]:
    historical_constraint = anchor.get("historical_constraint") or {}
    return (
        constraint_level_priority(historical_constraint.get("level")),
        float(anchor.get("retrieval_score", 0.0) or 0.0),
    )


def select_conflict_anchors(
    candidate_anchors: List[Dict[str, Any]],
    selected_pair_limit: int = MEMORY_SELECTED_PAIR_LIMIT,
    min_score: float = MEMORY_RETRIEVAL_MIN_SCORE,
    dedup_by_dimension: bool = MEMORY_DEDUP_BY_DIMENSION,
) -> List[Dict[str, Any]]:
    """Filter, deduplicate, and globally select conflict anchors for prompting."""
    selected: List[Dict[str, Any]] = []
    seen_memory_keys = set()
    seen_dimensions = set()

    sorted_pairs = sorted(
        candidate_anchors,
        key=conflict_anchor_sort_key,
        reverse=True,
    )
    for pair in sorted_pairs:
        score = float(pair.get("retrieval_score", 0.0) or 0.0)
        if score < min_score:
            continue

        memory_key = normalize_dedup_key(pair.get("memory_key"))
        if memory_key and memory_key in seen_memory_keys:
            continue

        historical_constraint = pair.get("historical_constraint") or {}
        dimension = normalize_dedup_key(historical_constraint.get("dimension"))
        if dedup_by_dimension and dimension and dimension in seen_dimensions:
            continue

        selected.append(pair)
        if memory_key:
            seen_memory_keys.add(memory_key)
        if dimension:
            seen_dimensions.add(dimension)
        if len(selected) >= selected_pair_limit:
            break

    return selected


def select_memory_pairs(
    candidate_pairs: List[Dict[str, Any]],
    selected_pair_limit: int = MEMORY_SELECTED_PAIR_LIMIT,
    min_score: float = MEMORY_RETRIEVAL_MIN_SCORE,
    dedup_by_dimension: bool = MEMORY_DEDUP_BY_DIMENSION,
) -> List[Dict[str, Any]]:
    return select_conflict_anchors(
        candidate_pairs,
        selected_pair_limit=selected_pair_limit,
        min_score=min_score,
        dedup_by_dimension=dedup_by_dimension,
    )


def retrieve_conflict_anchors(
    llm_client: OpenAI,
    model: str,
    constraints: Dict[str, Any],
    dialogue_text: str = "",
    memory_patterns: Optional[List[Dict[str, Any]]] = None,
    top_k_per_constraint: int = MEMORY_RETRIEVAL_TOP_K_PER_CONSTRAINT,
    selected_pair_limit: int = MEMORY_SELECTED_PAIR_LIMIT,
    min_score: float = MEMORY_RETRIEVAL_MIN_SCORE,
    dedup_by_dimension: bool = MEMORY_DEDUP_BY_DIMENSION,
) -> List[Dict[str, Any]]:
    """Retrieve selected conflict anchors by asking the LLM to match constraints to patterns."""
    constraints = get_active_constraints(constraints)
    constraints = [
        constraint
        for constraint in constraints
        if str(constraint.get("dimension", "")).strip()
        and str(constraint.get("value", "")).strip()
    ]
    if not constraints:
        return []

    patterns = memory_patterns if memory_patterns is not None else load_memory_patterns()
    if not patterns:
        return []

    prompt_constraints = [
        {
            "constraint_id": str(idx),
            **minimal_constraint_for_memory(constraint),
        }
        for idx, constraint in enumerate(constraints)
    ]
    prompt_patterns = [pattern_for_retrieval_prompt(item) for item in patterns]

    prompt = render_prompt(
        LLM_MEMORY_RETRIEVAL_PROMPT,
        TOP_K_PER_CONSTRAINT=str(top_k_per_constraint),
        SELECTED_PAIR_LIMIT=str(selected_pair_limit),
        DIALOGUE=dialogue_text,
        HISTORICAL_CONSTRAINTS=safe_json_dumps(prompt_constraints),
        MEMORY_PATTERNS=safe_json_dumps(prompt_patterns),
    )
    raw = call_llm_json(
        llm_client,
        model,
        prompt,
        operation="retrieve_conflict_anchors_llm",
        temperature=0.0,
        max_tokens=1200,
    )
    if raw.get("_error"):
        return []

    raw_anchors = raw.get("anchors")
    if not isinstance(raw_anchors, list):
        return []

    constraints_by_id = {
        str(idx): constraint for idx, constraint in enumerate(constraints)
    }
    patterns_by_pattern_id = {
        str(item.get("pattern_id", "")).strip(): item for item in patterns
    }

    candidate_anchors: List[Dict[str, Any]] = []
    per_constraint_counts: Dict[str, int] = {}
    for raw_anchor in raw_anchors:
        if not isinstance(raw_anchor, dict):
            continue
        constraint_id = str(raw_anchor.get("constraint_id", "")).strip()
        pattern_id = str(
            raw_anchor.get("pattern_id", raw_anchor.get("memory_key", ""))
        ).strip()
        if (
            constraint_id not in constraints_by_id
            or pattern_id not in patterns_by_pattern_id
        ):
            continue
        count = per_constraint_counts.get(constraint_id, 0)
        if count >= top_k_per_constraint:
            continue
        try:
            score = float(raw_anchor.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        constraint = constraints_by_id[constraint_id]
        memory_item = patterns_by_pattern_id[pattern_id]
        memory_key = str(memory_item.get("memory_key", "")).strip()
        candidate_anchors.append(
            {
                "retrieval_score": score,
                "retrieval_reason": raw_anchor.get("reason", ""),
                "memory_key": memory_key,
                "historical_constraint": minimal_constraint_for_memory(constraint),
                "memory_pattern": minimal_pattern_for_memory(memory_item),
            }
        )
        per_constraint_counts[constraint_id] = count + 1

    return select_conflict_anchors(
        candidate_anchors,
        selected_pair_limit=selected_pair_limit,
        min_score=min_score,
        dedup_by_dimension=dedup_by_dimension,
    )


def retrieve_memory_pairs(
    llm_client: OpenAI,
    model: str,
    constraints: Dict[str, Any],
    dialogue_text: str = "",
    memory_patterns: Optional[List[Dict[str, Any]]] = None,
    top_k_per_constraint: int = MEMORY_RETRIEVAL_TOP_K_PER_CONSTRAINT,
    selected_pair_limit: int = MEMORY_SELECTED_PAIR_LIMIT,
    min_score: float = MEMORY_RETRIEVAL_MIN_SCORE,
    dedup_by_dimension: bool = MEMORY_DEDUP_BY_DIMENSION,
) -> List[Dict[str, Any]]:
    return retrieve_conflict_anchors(
        llm_client,
        model,
        constraints,
        dialogue_text=dialogue_text,
        memory_patterns=memory_patterns,
        top_k_per_constraint=top_k_per_constraint,
        selected_pair_limit=selected_pair_limit,
        min_score=min_score,
        dedup_by_dimension=dedup_by_dimension,
    )


def compact_prompt_mapping(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if value is not None and str(value).strip()
    }


def conflict_anchor_for_prompt(anchor: Dict[str, Any], anchor_id: int) -> Dict[str, Any]:
    """Keep only conflict-anchor fields needed by the generator prompt."""
    historical_constraint = anchor.get("historical_constraint") or {}
    memory_pattern = anchor.get("memory_pattern") or anchor.get("conflict_pattern") or {}

    return {
        "anchor_id": str(anchor_id),
        "historical_constraint": compact_prompt_mapping(
            {
                "source": historical_constraint.get("source", ""),
                "level": historical_constraint.get("level", ""),
                "dimension": historical_constraint.get("dimension", ""),
                "value": historical_constraint.get("value", ""),
                "evidence_span": historical_constraint.get("evidence_span", ""),
            }
        ),
        "conflict_pattern": compact_prompt_mapping(
            {
                "pattern_id": memory_pattern.get("pattern_id", ""),
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
                "positive_trajectories": memory_pattern.get("positive_trajectories", []),
                "negative_trajectories": memory_pattern.get("negative_trajectories", []),
            }
        ),
    }


def selected_conflict_anchors_to_text(conflict_anchors: List[Dict[str, Any]]) -> str:
    """Render selected conflict anchors for the implicit-conflict prompt."""
    prompt_anchors = []
    for idx, anchor in enumerate(conflict_anchors):
        prompt_anchors.append(conflict_anchor_for_prompt(anchor, idx))
    return yaml.safe_dump(
        prompt_anchors,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()


def selected_memory_pairs_to_text(memory_pairs: List[Dict[str, Any]]) -> str:
    return selected_conflict_anchors_to_text(memory_pairs)
