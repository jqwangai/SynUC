"""Shared utilities for converting synthesized samples into training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


VALID_LABELS = ("Normal Dialogue", "Explicit Revision", "Implicit Conflict")
VALID_LABEL_SET = set(VALID_LABELS)
DEFAULT_MIN_CONFIDENCE = {
    "Normal Dialogue": 95,
    "Explicit Revision": 100,
    "Implicit Conflict": 90,
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read dictionary records from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[Warning] Skipped invalid JSON on line {line_number}: {exc}")
                continue
            if not isinstance(record, dict):
                print(f"[Warning] Skipped non-object record on line {line_number}.")
                continue
            records.append(record)
    return records


def write_records(
    path: Path,
    records: Sequence[Dict[str, Any]],
    output_format: str,
) -> None:
    """Write records as a JSON array or JSONL."""
    if output_format not in {"json", "jsonl"}:
        raise ValueError("output_format must be 'json' or 'jsonl'.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        if output_format == "json":
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
            return
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def truncate_assistant_content(
    content: str,
    enabled: bool = True,
    head_chars: int = 40,
    tail_chars: int = 40,
    min_chars: int = 120,
) -> str:
    """Compact long assistant turns while retaining their boundaries."""
    if not enabled or len(content) <= min_chars:
        return content

    head = content[:head_chars]
    tail = content[-tail_chars:] if tail_chars else ""
    omitted_chars = len(content) - len(head) - len(tail)
    if omitted_chars <= 0:
        return content
    return f"{head}...[{omitted_chars} characters omitted]...{tail}"


def messages_to_text(
    messages: Iterable[Dict[str, Any]],
    truncate_assistant: bool = True,
    head_chars: int = 40,
    tail_chars: int = 40,
    min_chars: int = 120,
) -> str:
    """Render structured dialogue messages as numbered plain text."""
    lines: List[str] = []
    user_count = 0
    assistant_count = 0

    for message in messages:
        role = str(message.get("role", "") or "")
        content = str(message.get("content", "") or "")
        if role == "user":
            user_count += 1
            prefix = f"U{user_count}"
        elif role == "assistant":
            assistant_count += 1
            prefix = f"A{assistant_count}"
            content = truncate_assistant_content(
                content,
                enabled=truncate_assistant,
                head_chars=head_chars,
                tail_chars=tail_chars,
                min_chars=min_chars,
            )
        elif role == "system":
            prefix = "system"
        else:
            prefix = role or "unknown role"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def normalize_label(value: Any) -> str:
    label = str(value or "").strip()
    return label if label in VALID_LABEL_SET else ""


def normalize_confidence(value: Any) -> Optional[int]:
    try:
        confidence = int(value)
    except (TypeError, ValueError):
        return None
    return confidence if 0 <= confidence <= 100 else None


def passes_confidence_threshold(
    label: str,
    confidence: Optional[int],
    thresholds: Mapping[str, int],
) -> bool:
    threshold = int(thresholds.get(label, 0))
    return threshold <= 0 or (confidence is not None and confidence >= threshold)


def get_llm_outputs(item: Dict[str, Any]) -> Dict[str, Any]:
    outputs = (item.get("meta") or {}).get("llm_outputs") or {}
    return outputs if isinstance(outputs, dict) else {}


def candidate_id_of(value: Any) -> str:
    return str(value if value is not None else "0")


def get_verify_results(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect verification results and deduplicate them by candidate ID."""
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    candidates = item.get("candidate_queries") or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            verification = candidate.get("verification")
            if not isinstance(verification, dict):
                continue
            result = dict(verification)
            candidate_id = candidate_id_of(
                result.get("candidate_id", candidate.get("candidate_id"))
            )
            result["candidate_id"] = candidate_id
            seen.add(candidate_id)
            results.append(result)

    verification = get_llm_outputs(item).get("verify") or {}
    raw_results = verification.get("results") if isinstance(verification, dict) else None
    if isinstance(raw_results, list):
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            result = dict(raw_result)
            candidate_id = candidate_id_of(result.get("candidate_id"))
            if candidate_id in seen:
                continue
            result["candidate_id"] = candidate_id
            seen.add(candidate_id)
            results.append(result)
    return results


def get_candidate_query(item: Dict[str, Any], candidate_id: Any) -> Optional[str]:
    target_id = candidate_id_of(candidate_id)
    candidates = item.get("candidate_queries") or []
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate_id_of(candidate.get("candidate_id")) != target_id:
            continue
        query = (
            candidate.get("current_query")
            or candidate.get("new_query")
            or candidate.get("query")
        )
        if isinstance(query, dict):
            query = query.get("new_query") or query.get("current_query")
        return str(query).strip() if query is not None else None
    return None


def get_current_query(
    item: Dict[str, Any],
    verify_result: Dict[str, Any],
) -> Optional[str]:
    label = item.get("label")
    if label in {"Normal Dialogue", "Explicit Revision"}:
        query = item.get("current_query")
        return str(query).strip() if query is not None else None
    if label == "Implicit Conflict":
        return get_candidate_query(item, verify_result.get("candidate_id"))
    return None


def should_skip_item(item: Dict[str, Any]) -> bool:
    if item.get("label") == "Reject":
        return True
    return (item.get("meta") or {}).get("is_training_sample") is False


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def format_value_or_none(value: Any) -> str:
    text = normalize_text(value)
    return text if text else "None"


def get_nested_task_field(
    features: Dict[str, Any],
    direct_key: str,
    nested_key: str,
) -> Any:
    if direct_key in features:
        return features.get(direct_key)
    current_main_task = features.get("current_main_task")
    if isinstance(current_main_task, dict):
        return current_main_task.get(nested_key)
    return None


def task_goal_text(value: Any) -> str:
    if not value:
        return "None"
    if isinstance(value, dict):
        return format_value_or_none(
            value.get("main_intent_goal")
            or value.get("sub_intent_goal")
            or value.get("intent_goal")
        )
    return format_value_or_none(value)


def task_goal_list_text(value: Any) -> str:
    if not value:
        return "None"
    if not isinstance(value, list):
        return task_goal_text(value)
    goals = [task_goal_text(item) for item in value]
    goals = [goal for goal in goals if goal and goal != "None"]
    return "; ".join(goals) if goals else "None"


def constraint_text(constraint: Any) -> str:
    if not isinstance(constraint, dict):
        return format_value_or_none(constraint)
    dimension = format_value_or_none(constraint.get("dimension"))
    value = format_value_or_none(constraint.get("value"))
    source = normalize_text(constraint.get("source"))
    evidence_span = normalize_text(constraint.get("evidence_span"))
    if source == "implicit" or evidence_span:
        if evidence_span:
            return f"Implicit Constraint: infer '{dimension}: {value}' from '{evidence_span}'"
        return f"Implicit Constraint: '{dimension}: {value}'"
    return f"Explicit Constraint: '{dimension}: {value}'"


def constraints_text(constraints: Any) -> str:
    if not constraints:
        return "None"
    if not isinstance(constraints, list):
        return constraint_text(constraints)
    items = [constraint_text(constraint) for constraint in constraints]
    items = [item for item in items if item and item != "None"]
    return "; ".join(items) if items else "None"
