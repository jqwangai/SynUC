"""Trajectory-buffer persistence for implicit-conflict verification feedback."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .memory_ops import call_llm_json, render_prompt, safe_json_dumps


POSITIVE_TRAJECTORIES = "positive_trajectories"
NEGATIVE_TRAJECTORIES = "negative_trajectories"
MEMORY_TEXT_FIELDS = {
    "definition",
    "description",
    "retrieval_terms",
    "historical_constraint_feature",
    "new_constraint_feature",
    "conflict_focus",
}


POSITIVE_TRAJECTORY_FUSION_PROMPT = """
You are an implicit-conflict positive trajectory fusion tool. Based on the pattern definition, deduplicate and quality-filter the candidate positive trajectory pool in a unified way.

## Judgement Objective
A positive trajectory must be usable as a successful construction experience for this pattern: historical_constraint and new_constraint match the pattern's historical_constraint_feature and new_constraint_feature, the two constraints truly cannot be satisfied simultaneously, user_query_with_new_constraint can naturally carry new_constraint, and the trajectory does not obviously write an implicit conflict as an explicit revision.

## Selection Rules
1. Select only from the IDs provided in the input. Do not rewrite, synthesize, or add trajectories.
2. For trajectories that are semantically duplicate, differ only in wording, or follow highly similar construction paths, keep only the highest-quality, most specific, and most transferable one.
3. Remove trajectories that do not form a real conflict, are only extensions/refinements/supplements, or are inconsistent with this pattern's conflict_focus.
4. Prefer trajectories that cover different contexts, constraint dimensions, or naturalization methods.
5. Each element in the output array represents one positive trajectory that should be retained; selected_id should contain only the ID of that trajectory.
6. Select at most <<MAX_TRAJECTORIES>> positive trajectories. If the number of valid trajectories exceeds the limit, keep only the most representative trajectories that can best guide subsequent generation.

All text output must be in English.

Output only a valid JSON array. Do not output explanatory text. Output format:
[
  {
    "selection_reason": "Explain why P0 is retained, or why it is better than duplicate alternatives",
    "selected_id": ["P0"]
  },
  {
    "selection_reason": "Explain why P2 is retained, or what different experience it covers",
    "selected_id": ["P2"]
  }
]

Pattern:
<<PATTERN_CONTEXT>>

Candidate positive trajectories:
<<CANDIDATE_TRAJECTORIES>>
""".strip()


NEGATIVE_TRAJECTORY_FUSION_PROMPT = """
You are an implicit-conflict negative trajectory fusion tool. Based on the pattern definition, deduplicate and quality-filter the candidate negative trajectory pool in a unified way, and output a one-sentence summarized rejection_reason.

## Judgement Objective
A negative trajectory must be usable as a failed construction experience for this pattern: it appears relevant to this pattern, but the verification result shows that it should not be used as a positive implicit-conflict example. For example, the two constraints can be satisfied simultaneously, the query is only an ordinary supplement/refinement/extension, the continuation level is wrong, new_constraint is exposed too explicitly, or the trajectory is inconsistent with conflict_focus.

## Selection Rules
1. Select only from the IDs provided in the input. Do not rewrite, synthesize, or add trajectories.
2. For negative trajectories with duplicate semantics, the same failure reason, and similar contexts, keep only the one with the highest diagnostic value.
3. Remove negative trajectories that are unrelated to this pattern, have unclear rejection reasons, or cannot guide later generation to avoid errors.
4. Prefer negative trajectories that cover different failure types.
5. Each element in the output array represents one negative trajectory that should be retained; selected_id should contain only the ID of that trajectory.
6. Select at most <<MAX_TRAJECTORIES>> negative trajectories. If the number of valid trajectories exceeds the limit, keep only the most representative trajectories that can best guide later error avoidance.
7. rejection_reason must be one concise English sentence summarizing the main failure reason reflected by the negative trajectory. If there are no valid negative trajectories, output an empty array [].

All text output must be in English.

Output only a valid JSON array. Do not output explanatory text. Output format:
[
  {
    "selection_reason": "Explain why N0 is retained, or what type of failed construction experience it reflects",
    "selected_id": ["N0"],
    "rejection_reason": "Summarize the failure reason of N0 in one sentence"
  },
  {
    "selection_reason": "Explain why N2 is retained, or what different failed construction experience it covers",
    "selected_id": ["N2"],
    "rejection_reason": "Summarize the failure reason of N2 in one sentence"
  }
]

Pattern:
<<PATTERN_CONTEXT>>

Candidate negative trajectories:
<<CANDIDATE_TRAJECTORIES>>
""".strip()


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def empty_buffer(threshold: int) -> Dict[str, Any]:
    return {
        "threshold": threshold,
        "updated_at": now_text(),
        "patterns": {},
    }


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def atomic_write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    os.replace(tmp_path, path)


def normalize_memory_text_fields(memory: Dict[str, Any]) -> None:
    """Trim YAML-folded metadata strings before rewriting the memory file."""
    operators = memory.get("operators")
    if not isinstance(operators, list):
        return
    for operator in operators:
        if not isinstance(operator, dict):
            continue
        for key in MEMORY_TEXT_FIELDS:
            value = operator.get(key)
            if isinstance(value, str):
                operator[key] = value.strip()
        patterns = operator.get("patterns")
        if not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            for key in MEMORY_TEXT_FIELDS:
                value = pattern.get(key)
                if isinstance(value, str):
                    pattern[key] = value.strip()


def trajectory_identity(trajectory: Dict[str, Any]) -> str:
    identity = {
        "context": trajectory.get("context", ""),
        "historical_constraint": trajectory.get("historical_constraint", {}),
        "new_constraint": trajectory.get("new_constraint", {}),
        "user_query_with_new_constraint": trajectory.get(
            "user_query_with_new_constraint", ""
        ),
        "source_id": trajectory.get("source_id", ""),
        "candidate_id": trajectory.get("candidate_id", ""),
    }
    return json.dumps(identity, ensure_ascii=False, sort_keys=True)


def minimal_constraint(constraint: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": constraint.get("source", ""),
        "level": constraint.get("level", ""),
        "dimension": constraint.get("dimension", ""),
        "value": constraint.get("value", ""),
    }


def context_from_constraints(
    constraints_doc: Dict[str, Any],
    new_constraint: Dict[str, Any],
) -> str:
    current_main_task = constraints_doc.get("current_main_task") or {}
    if not isinstance(current_main_task, dict):
        return ""

    main_goal = str(current_main_task.get("main_intent_goal", "") or "").strip()
    current_subtask = current_main_task.get("current_subtask") or {}
    sub_goal = ""
    if isinstance(current_subtask, dict):
        sub_goal = str(current_subtask.get("sub_intent_goal", "") or "").strip()

    level = str(new_constraint.get("level", "") or "").strip()
    if level == "current_subtask":
        return "; ".join(part for part in (main_goal, sub_goal) if part)
    return main_goal


def verification_rejection_reason(verification: Dict[str, Any]) -> str:
    for key in ("reasoning", "task_judge", "reason", "explanation"):
        value = verification.get(key)
        if value:
            return str(value)
    label = verification.get("label")
    return f"verification label is {label}" if label else ""


def compact_trajectory_for_prompt(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Remove bookkeeping fields before sending a trajectory to the fusion LLM."""
    compact = {
        "context": trajectory.get("context", ""),
        "historical_constraint": compact_constraint_for_prompt(
            trajectory.get("historical_constraint")
        ),
        "new_constraint": compact_constraint_for_prompt(
            trajectory.get("new_constraint")
        ),
        "user_query_with_new_constraint": trajectory.get(
            "user_query_with_new_constraint", ""
        ),
    }
    rejection_reason = trajectory.get("rejection_reason")
    if rejection_reason:
        compact["rejection_reason"] = rejection_reason
    return {
        key: value
        for key, value in compact.items()
        if value is not None and str(value).strip()
    }


def trajectory_for_memory(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Remove run-local bookkeeping fields before storing trajectories in memory."""
    return {
        key: value
        for key, value in trajectory.items()
        if key not in {"source_id", "candidate_id", "verified_at"}
    }


def compact_constraint_for_prompt(constraint: Any) -> Dict[str, Any]:
    if not isinstance(constraint, dict):
        return {}
    return {
        key: constraint.get(key, "")
        for key in ("dimension", "value")
        if str(constraint.get(key, "")).strip()
    }


class TrajectoryBuffer:
    """Accumulate verified trajectories and promote mature patterns to memory."""

    def __init__(
        self,
        buffer_path: Path,
        memory_path: Path,
        threshold: int = 5,
        llm_client: Optional[Any] = None,
        model: Optional[str] = None,
        max_positive_trajectories: int = 3,
        max_negative_trajectories: int = 3,
    ) -> None:
        self.buffer_path = buffer_path
        self.memory_path = memory_path
        self.threshold = threshold
        self.llm_client = llm_client
        self.model = model
        self.max_positive_trajectories = max(0, max_positive_trajectories)
        self.max_negative_trajectories = max(0, max_negative_trajectories)
        self.data = self._load_buffer()

    def _load_buffer(self) -> Dict[str, Any]:
        data = load_yaml_mapping(self.buffer_path)
        if not data:
            return empty_buffer(self.threshold)
        patterns = data.get("patterns")
        if not isinstance(patterns, dict):
            data["patterns"] = {}
        data["threshold"] = data.get("threshold", self.threshold)
        data["updated_at"] = data.get("updated_at", now_text())
        return data

    def add_events(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        stats = {
            "added": 0,
            "promoted": 0,
            "skipped": 0,
        }
        for event in events:
            pattern_id = str(event.get("pattern_id", "") or "").strip()
            trajectory_type = str(event.get("trajectory_type", "") or "").strip()
            trajectory = event.get("trajectory")
            if (
                not pattern_id
                or pattern_id == "no_memory"
                or trajectory_type not in ("positive", "negative")
                or not isinstance(trajectory, dict)
            ):
                stats["skipped"] += 1
                continue
            if self._add_trajectory(pattern_id, trajectory_type, trajectory):
                stats["added"] += 1
            else:
                stats["skipped"] += 1

        promoted = self.promote_ready_patterns()
        stats["promoted"] = promoted
        self.data["updated_at"] = now_text()
        atomic_write_yaml(self.buffer_path, self.data)
        return stats

    def _add_trajectory(
        self,
        pattern_id: str,
        trajectory_type: str,
        trajectory: Dict[str, Any],
    ) -> bool:
        patterns = self.data.setdefault("patterns", {})
        pattern = patterns.setdefault(
            pattern_id,
            {POSITIVE_TRAJECTORIES: [], NEGATIVE_TRAJECTORIES: []},
        )
        positive = pattern.setdefault(POSITIVE_TRAJECTORIES, [])
        negative = pattern.setdefault(NEGATIVE_TRAJECTORIES, [])
        target_key = (
            POSITIVE_TRAJECTORIES
            if trajectory_type == "positive"
            else NEGATIVE_TRAJECTORIES
        )
        target = pattern.setdefault(target_key, [])
        existing_ids = {
            trajectory_identity(item)
            for item in positive + negative
            if isinstance(item, dict)
        }
        if trajectory_identity(trajectory) in existing_ids:
            return False
        target.append(trajectory)
        return True

    def promote_ready_patterns(self) -> int:
        patterns = self.data.get("patterns")
        if not isinstance(patterns, dict):
            return 0

        ready = [
            (pattern_id, ready_fields)
            for pattern_id, pattern in patterns.items()
            for ready_fields in [self._ready_trajectory_fields(pattern)]
            if ready_fields
        ]
        if not ready:
            return 0

        memory = load_yaml_mapping(self.memory_path)
        promoted = 0
        for pattern_id, ready_fields in ready:
            trajectory_group = patterns.get(pattern_id)
            if not isinstance(trajectory_group, dict):
                continue
            if self._merge_pattern_into_memory(
                memory,
                pattern_id,
                trajectory_group,
                ready_fields,
            ):
                for field in ready_fields:
                    trajectory_group[field] = []
                if self._is_empty_trajectory_group(trajectory_group):
                    del patterns[pattern_id]
                promoted += 1

        if promoted:
            normalize_memory_text_fields(memory)
            atomic_write_yaml(self.memory_path, memory)
        return promoted

    def _ready_trajectory_fields(self, pattern: Any) -> List[str]:
        if not isinstance(pattern, dict):
            return []
        ready = []
        for field in (POSITIVE_TRAJECTORIES, NEGATIVE_TRAJECTORIES):
            items = pattern.get(field)
            count = len(items) if isinstance(items, list) else 0
            if count >= self.threshold:
                ready.append(field)
        return ready

    def _is_empty_trajectory_group(self, pattern: Dict[str, Any]) -> bool:
        positive = pattern.get(POSITIVE_TRAJECTORIES)
        negative = pattern.get(NEGATIVE_TRAJECTORIES)
        positive_count = len(positive) if isinstance(positive, list) else 0
        negative_count = len(negative) if isinstance(negative, list) else 0
        return positive_count == 0 and negative_count == 0

    def _merge_pattern_into_memory(
        self,
        memory: Dict[str, Any],
        pattern_id: str,
        trajectory_group: Dict[str, Any],
        ready_fields: List[str],
    ) -> bool:
        target = self._find_memory_pattern(memory, pattern_id)
        if target is None:
            return False
        if self.llm_client is None or not self.model:
            print(
                f"[TrajectoryBuffer] skip promotion for {pattern_id}: "
                "LLM client/model is required for trajectory fusion"
            )
            return False

        positive_source = trajectory_group.get(POSITIVE_TRAJECTORIES)
        negative_source = trajectory_group.get(NEGATIVE_TRAJECTORIES)

        fused_positive: Optional[List[Dict[str, Any]]] = None
        fused_negative: Optional[Dict[str, Any]] = None

        if (
            POSITIVE_TRAJECTORIES in ready_fields
            and isinstance(positive_source, list)
            and positive_source
        ):
            fused_positive = self._fuse_positive_trajectories(
                pattern_id,
                target,
                positive_source,
            )
            if fused_positive is None:
                return False

        if (
            NEGATIVE_TRAJECTORIES in ready_fields
            and isinstance(negative_source, list)
            and negative_source
        ):
            fused_negative = self._fuse_negative_trajectories(
                pattern_id,
                target,
                negative_source,
            )
            if fused_negative is None:
                return False

        if fused_positive is not None:
            target[POSITIVE_TRAJECTORIES] = [
                trajectory_for_memory(item) for item in fused_positive
            ]
        if fused_negative is not None:
            target[NEGATIVE_TRAJECTORIES] = [
                trajectory_for_memory(item)
                for item in fused_negative["trajectories"]
            ]
            rejection_reason = str(fused_negative.get("rejection_reason", "")).strip()
            if rejection_reason:
                target["rejection_reason"] = rejection_reason
            else:
                target.pop("rejection_reason", None)
        return True

    def _fuse_positive_trajectories(
        self,
        pattern_id: str,
        target: Dict[str, Any],
        buffer_items: List[Any],
    ) -> Optional[List[Dict[str, Any]]]:
        existing_items = self._valid_trajectory_list(
            target.get(POSITIVE_TRAJECTORIES)
        )
        buffer_items = self._valid_trajectory_list(buffer_items)
        numbered = self._numbered_trajectories("P", existing_items, buffer_items)
        if not numbered:
            return []

        pattern_context = self._pattern_context_for_prompt(pattern_id, target)
        prompt = render_prompt(
            POSITIVE_TRAJECTORY_FUSION_PROMPT,
            MAX_TRAJECTORIES=str(self.max_positive_trajectories),
            PATTERN_CONTEXT=safe_json_dumps(pattern_context),
            CANDIDATE_TRAJECTORIES=safe_json_dumps(
                self._numbered_prompt_trajectories(numbered)
            ),
        )
        raw = call_llm_json(
            self.llm_client,
            self.model or "",
            prompt,
            operation="fuse_positive_trajectories",
            temperature=0.0,
            max_tokens=1200,
            json_object_response=False,
        )
        if isinstance(raw, dict) and raw.get("_error"):
            return None

        selections = self._selection_results_from_llm(raw)
        if selections is None:
            return None
        selected_ids = self._selected_ids_from_llm(selections)
        if selected_ids is None:
            return None
        selected_ids = selected_ids[: self.max_positive_trajectories]
        return self._items_from_selected_ids(
            numbered,
            selected_ids,
            self.max_positive_trajectories,
        )

    def _fuse_negative_trajectories(
        self,
        pattern_id: str,
        target: Dict[str, Any],
        buffer_items: List[Any],
    ) -> Optional[Dict[str, Any]]:
        existing_items = self._valid_trajectory_list(
            target.get(NEGATIVE_TRAJECTORIES)
        )
        buffer_items = self._valid_trajectory_list(buffer_items)
        numbered = self._numbered_trajectories("N", existing_items, buffer_items)
        if not numbered:
            return {"trajectories": [], "rejection_reason": ""}

        pattern_context = self._pattern_context_for_prompt(pattern_id, target)
        prompt = render_prompt(
            NEGATIVE_TRAJECTORY_FUSION_PROMPT,
            MAX_TRAJECTORIES=str(self.max_negative_trajectories),
            PATTERN_CONTEXT=safe_json_dumps(pattern_context),
            CANDIDATE_TRAJECTORIES=safe_json_dumps(
                self._numbered_prompt_trajectories(numbered)
            ),
        )
        raw = call_llm_json(
            self.llm_client,
            self.model or "",
            prompt,
            operation="fuse_negative_trajectories",
            temperature=0.0,
            max_tokens=1400,
            json_object_response=False,
        )
        if isinstance(raw, dict) and raw.get("_error"):
            return None

        selections = self._selection_results_from_llm(raw)
        if selections is None:
            return None
        selected_ids = self._selected_ids_from_llm(selections)
        if selected_ids is None:
            return None
        selected_ids = selected_ids[: self.max_negative_trajectories]
        selected_items = self._items_from_selected_ids(
            numbered,
            selected_ids,
            self.max_negative_trajectories,
        )
        return {
            "trajectories": selected_items,
            "rejection_reason": self._rejection_reason_from_trajectories(
                selected_items
            ),
        }

    def _pattern_context_for_prompt(
        self,
        pattern_id: str,
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "pattern_id": pattern_id,
            "description": target.get("description", ""),
            "historical_constraint_feature": target.get(
                "historical_constraint_feature",
                target.get("historical_constraint", ""),
            ),
            "new_constraint_feature": target.get(
                "new_constraint_feature",
                target.get("new_constraint", ""),
            ),
            "conflict_focus": target.get("conflict_focus", ""),
        }

    def _valid_trajectory_list(self, items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def _numbered_trajectories(
        self,
        prefix: str,
        memory_items: List[Dict[str, Any]],
        buffer_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        numbered: List[Dict[str, Any]] = []
        index = 0
        for source, items in (("memory", memory_items), ("buffer", buffer_items)):
            for item in items:
                numbered.append(
                    {
                        "id": f"{prefix}{index}",
                        "source": source,
                        "item": item,
                    }
                )
                index += 1
        return numbered

    def _numbered_prompt_trajectories(
        self,
        numbered: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": item["id"],
                "trajectory": compact_trajectory_for_prompt(item["item"]),
            }
            for item in numbered
        ]

    def _selection_results_from_llm(
        self,
        raw: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        if isinstance(raw, list):
            items = [item for item in raw if isinstance(item, dict)]
            return items if items else None
        if isinstance(raw, dict):
            for key in ("selections", "results", "items"):
                value = raw.get(key)
                if isinstance(value, list):
                    items = [item for item in value if isinstance(item, dict)]
                    return items if items else None
            return [raw]
        return None

    def _selected_ids_from_llm(
        self,
        selections: List[Dict[str, Any]],
    ) -> Optional[List[str]]:
        selected_ids: List[str] = []
        for selection in selections:
            selected = (
                selection.get("selected_id")
                or selection.get("selected_ids")
                or selection.get("selected_trajectory_id")
                or selection.get("selected_trajectory_ids")
                or selection.get("id")
                or selection.get("ids")
                or selection.get("selected_trajectories")
            )
            if selected is None:
                continue
            if not isinstance(selected, list):
                selected = [selected]
            for item in selected:
                if isinstance(item, dict):
                    item = item.get("id", item.get("trajectory_id", ""))
                selected_id = str(item).strip()
                if selected_id:
                    selected_ids.append(selected_id)
        return selected_ids if selected_ids else None

    def _rejection_reason_from_llm(
        self,
        selections: List[Dict[str, Any]],
        selected_ids: List[str],
    ) -> str:
        reasons = []
        for selection in selections:
            selection_ids = self._selected_ids_from_llm([selection]) or []
            if not any(selected_id in selected_ids for selected_id in selection_ids):
                continue
            reason = str(selection.get("rejection_reason", "") or "").strip()
            if reason and reason not in reasons:
                reasons.append(reason)
        return "; ".join(reasons)

    def _rejection_reason_from_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> str:
        reasons = []
        for trajectory in trajectories:
            reason = str(trajectory.get("rejection_reason", "") or "").strip()
            if reason and reason not in reasons:
                reasons.append(reason)
        return "; ".join(reasons)

    def _items_from_selected_ids(
        self,
        numbered: List[Dict[str, Any]],
        selected_ids: List[str],
        max_items: int,
    ) -> List[Dict[str, Any]]:
        if max_items <= 0:
            return []
        item_by_id = {item["id"]: item["item"] for item in numbered}
        seen_identity = set()
        selected_items: List[Dict[str, Any]] = []
        for selected_id in selected_ids:
            item = item_by_id.get(selected_id)
            if item is None:
                continue
            identity = trajectory_identity(item)
            if identity in seen_identity:
                continue
            selected_items.append(item)
            seen_identity.add(identity)
            if len(selected_items) >= max_items:
                break
        return selected_items

    def _find_memory_pattern(
        self,
        memory: Dict[str, Any],
        pattern_id: str,
    ) -> Optional[Dict[str, Any]]:
        operators = memory.get("operators")
        if not isinstance(operators, list):
            return None
        for operator in operators:
            if not isinstance(operator, dict):
                continue
            patterns = operator.get("patterns")
            if not isinstance(patterns, list):
                continue
            for pattern in patterns:
                if (
                    isinstance(pattern, dict)
                    and str(pattern.get("pattern_id", "")).strip() == pattern_id
                ):
                    return pattern
        return None


def trajectory_buffer_event(
    pattern_id: str,
    trajectory_type: str,
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "pattern_id": pattern_id,
        "trajectory_type": trajectory_type,
        "trajectory": trajectory,
    }


def build_trajectory(
    source_id: str,
    candidate_id: str,
    context: str,
    historical_constraint: Dict[str, Any],
    new_constraint: Dict[str, Any],
    user_query: str,
    verification: Dict[str, Any],
    is_positive: bool,
) -> Dict[str, Any]:
    trajectory: Dict[str, Any] = {
        "context": context,
        "historical_constraint": minimal_constraint(historical_constraint),
        "new_constraint": minimal_constraint(new_constraint),
        "user_query_with_new_constraint": user_query,
        "source_id": source_id,
        "candidate_id": candidate_id,
        "verified_at": now_text(),
    }
    if not is_positive:
        trajectory["rejection_reason"] = verification_rejection_reason(verification)
    return trajectory


def summarize_stats(stats: Dict[str, int]) -> str:
    return (
        f"added={stats.get('added', 0)} | "
        f"promoted={stats.get('promoted', 0)} | "
        f"skipped={stats.get('skipped', 0)}"
    )
