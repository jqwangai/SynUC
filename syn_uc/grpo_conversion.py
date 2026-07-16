"""Convert synthesized Syn-UC samples into GRPO classification data.

Normal Dialogue and Explicit Revision records use the top-level current query.
Implicit Conflict records may contain several candidate queries, each linked to
its verification result by candidate ID. Only verified candidates whose label
matches the source record are eligible for training.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import conversion_utils as common
from . import sft_conversion
from .config import DATA_DIR, OUTPUT_PATH


DEFAULT_INPUT_PATH = Path(OUTPUT_PATH)
DEFAULT_OUTPUT_PATH = DATA_DIR / "training" / "grpo.jsonl"

VALID_LABELS = ["Normal Dialogue", "Explicit Revision", "Implicit Conflict"]
VALID_LABEL_SET = set(VALID_LABELS)

# Per-label sample fractions. For example, 0.4 means "keep 40% of this label".
LABEL_SAMPLE_WEIGHTS = {
    "Normal Dialogue": 1,
    "Explicit Revision": 0.33,
    "Implicit Conflict": 1,
}

DEFAULT_SEED = 42

# Only keep verified non-Reject samples whose confidence is at least the
# label-specific value below. Set a label's value to 0 to disable filtering for
# that label.
MIN_VERIFY_CONFIDENCE_BY_LABEL = {
    "Normal Dialogue": 95,
    "Explicit Revision": 100,
    "Implicit Conflict": 90,
}


@dataclass
class Candidate:
    row: Dict[str, Any]
    label: str
    source_id: Optional[str]
    source_index: int


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return common.read_jsonl(path)


def write_rows(
    path: Path,
    rows: List[Dict[str, Any]],
    output_format: str,
) -> None:
    common.write_records(path, rows, output_format)


def normalize_label(value: Any) -> str:
    label = str(value or "").strip()
    return label if label in VALID_LABEL_SET else ""


def normalize_confidence(value: Any) -> Optional[int]:
    try:
        confidence = int(value)
    except (TypeError, ValueError):
        return None

    if confidence < 0 or confidence > 100:
        return None
    return confidence


def simplify_constraint(constraint: Any) -> Dict[str, Any]:
    if not isinstance(constraint, dict):
        return {"source": "", "dimension": "", "value": constraint}
    return {
        "source": constraint.get("source", ""),
        "dimension": constraint.get("dimension", ""),
        "value": constraint.get("value"),
    }


def normalize_constraints(value: Any) -> List[Dict[str, Any]]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    return [constraint for constraint in values if isinstance(constraint, dict)]


def find_constraint_pair_by_candidate(
    pairs: Any,
    candidate_id: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(pairs, list):
        return None
    target_candidate_id = candidate_id_of(candidate_id)
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        if candidate_id_of(pair.get("candidate_id")) == target_candidate_id:
            return pair
    return None


def get_constructed_constraint_pair(
    item: Dict[str, Any],
    verify_result: Dict[str, Any],
    label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    outputs = get_llm_outputs(item)
    candidate_id = verify_result.get("candidate_id")

    if label == "Implicit Conflict":
        generation = outputs.get("generate_implicit") or {}
        pair = find_constraint_pair_by_candidate(
            generation.get("new_constraints_generation"),
            candidate_id,
        )
        if pair:
            return (
                simplify_constraint(pair.get("historical_constraint")),
                simplify_constraint(pair.get("new_constraint")),
            )

    if label == "Explicit Revision":
        generation = outputs.get("generate_explicit_revision") or {}
        generation = generation.get("generation") if isinstance(generation, dict) else {}
        pair = find_constraint_pair_by_candidate(
            generation.get("new_constraints") if isinstance(generation, dict) else None,
            candidate_id,
        )
        if pair:
            return (
                simplify_constraint(pair.get("historical_constraint")),
                simplify_constraint(pair.get("new_constraint")),
            )

    return get_verify_constraint_pair(verify_result)


def source_from_channel(channel: Any, index: int) -> str:
    parts = str(channel or "").split("_")
    if len(parts) >= 4:
        source_part = parts[index]
        if source_part == "E":
            return "explicit"
        if source_part == "I":
            return "implicit"
    return ""


def first_constraint_with_source(
    constraints: Any,
    source: str,
) -> Dict[str, Any]:
    normalized = normalize_constraints(constraints)
    if source:
        for constraint in normalized:
            if constraint.get("source") == source:
                return simplify_constraint(constraint)
    return simplify_constraint(normalized[0]) if normalized else {}


def get_verify_constraint_pair(
    verify_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    history_source = source_from_channel(verify_result.get("channel"), 1)
    new_source = source_from_channel(verify_result.get("channel"), 3)
    history_constraint = first_constraint_with_source(
        verify_result.get("history"),
        history_source,
    )

    new_explicit_constraints = (
        verify_result.get("new_explicit_constraints")
        or verify_result.get("current_explicit_constraints")
    )
    new_implicit_constraints = (
        verify_result.get("new_implicit_constraints")
        or verify_result.get("current_implicit_constraints")
    )
    new_constraints = (
        normalize_constraints(new_explicit_constraints)
        + normalize_constraints(new_implicit_constraints)
    )
    for constraint in new_constraints:
        constraint.setdefault("source", new_source)
    new_constraint = first_constraint_with_source(new_constraints, new_source)

    return history_constraint, new_constraint


def should_skip_source_item(item: Dict[str, Any]) -> bool:
    if item.get("label") == "Reject":
        return True
    return (item.get("meta") or {}).get("is_training_sample") is False


def get_llm_outputs(item: Dict[str, Any]) -> Dict[str, Any]:
    outputs = (item.get("meta") or {}).get("llm_outputs") or {}
    return outputs if isinstance(outputs, dict) else {}


def candidate_id_of(value: Any) -> str:
    return str(value if value is not None else "0")


def get_candidate_query(candidate: Dict[str, Any]) -> str:
    query = (
        candidate.get("current_query")
        or candidate.get("new_query")
        or candidate.get("query")
    )
    if isinstance(query, dict):
        query = query.get("new_query") or query.get("current_query")
    return str(query or "").strip()


def collect_implicit_candidate_queries(item: Dict[str, Any]) -> Dict[str, str]:
    """Collect follow-up queries from an aggregated implicit-conflict sample.

    item["candidate_queries"] = [
      {
        "candidate_id": "0",
        "current_query": "...",
        "verification": {...}
      },
      ...
    ]
    """
    queries: Dict[str, str] = {}
    candidates = item.get("candidate_queries") or []
    if not isinstance(candidates, list):
        return queries

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate_id_of(candidate.get("candidate_id"))
        query = get_candidate_query(candidate)
        if query:
            queries[candidate_id] = query
    return queries


def collect_verify_results(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read verification results and deduplicate them by candidate ID."""
    results: List[Dict[str, Any]] = []
    seen = set()

    candidates = item.get("candidate_queries") or []
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            verification = candidate.get("verification")
            if not isinstance(verification, dict):
                continue
            result = dict(verification)
            result["candidate_id"] = candidate_id_of(
                result.get("candidate_id", candidate.get("candidate_id"))
            )
            seen.add(result["candidate_id"])
            results.append(result)

    verify = get_llm_outputs(item).get("verify") or {}
    raw_results = verify.get("results") if isinstance(verify, dict) else None
    if isinstance(raw_results, list):
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            result = dict(raw)
            result["candidate_id"] = candidate_id_of(result.get("candidate_id"))
            if result["candidate_id"] in seen:
                continue
            seen.add(result["candidate_id"])
            results.append(result)

    return results


def build_prompt(item: Dict[str, Any], subsequent_query: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": sft_conversion.build_user_content(item, subsequent_query),
        }
    ]


def current_query_for(
    item: Dict[str, Any],
    verify_result: Dict[str, Any],
    implicit_queries: Dict[str, str],
) -> str:
    label = item.get("label")
    if label in {"Normal Dialogue", "Explicit Revision"}:
        return str(item.get("current_query") or "").strip()
    if label == "Implicit Conflict":
        return implicit_queries.get(candidate_id_of(verify_result.get("candidate_id")), "")
    return ""


def build_grpo_row(
    item: Dict[str, Any],
    verify_result: Dict[str, Any],
    implicit_queries: Dict[str, str],
    min_confidence_by_label: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    label = normalize_label(item.get("label"))
    if not label or verify_result.get("label") != label:
        return None

    confidence = normalize_confidence(verify_result.get("confidence"))
    if not common.passes_confidence_threshold(
        label,
        confidence,
        min_confidence_by_label,
    ):
        return None

    subsequent_query = current_query_for(item, verify_result, implicit_queries)
    if not subsequent_query:
        return None

    candidate_id = verify_result.get("candidate_id")
    row_id = str(item.get("sample_id") or item.get("source_id") or "")
    if label == "Implicit Conflict":
        row_id = f"{row_id}_candidate_{candidate_id_of(candidate_id)}"

    row = {
        "id": row_id,
        "prompt": build_prompt(item, subsequent_query),
        "ground_truth": label,
        "confidence": confidence,
    }
    if label != "Normal Dialogue":
        history_constraint, new_constraint = get_constructed_constraint_pair(
            item,
            verify_result,
            label,
        )
        row["ground_truth_history_constraint"] = history_constraint
        row["ground_truth_new_constraint"] = new_constraint
    return row


def to_ms_swift_row(
    row: Dict[str, Any],
    include_id: bool = False,
) -> Dict[str, Any]:
    messages = row.get("prompt") or row.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    ms_swift_row = {
        "messages": messages,
        "ground_truth": row.get("ground_truth"),
    }
    if row.get("ground_truth") != "Normal Dialogue":
        ms_swift_row["ground_truth_history_constraint"] = row.get(
            "ground_truth_history_constraint", {}
        )
        ms_swift_row["ground_truth_new_constraint"] = row.get(
            "ground_truth_new_constraint", {}
        )
    if include_id and row.get("id") is not None:
        ms_swift_row["id"] = row["id"]
    return ms_swift_row


def build_all_candidates(
    items: Iterable[Dict[str, Any]],
    min_confidence_by_label: Optional[Dict[str, int]] = None,
) -> Tuple[List[Candidate], Counter]:
    candidates: List[Candidate] = []
    skipped = Counter()
    thresholds = min_confidence_by_label or dict(MIN_VERIFY_CONFIDENCE_BY_LABEL)

    for source_index, item in enumerate(items):
        if should_skip_source_item(item):
            skipped["skip_source_item"] += 1
            continue

        label = normalize_label(item.get("label"))
        if not label:
            skipped["bad_label"] += 1
            continue

        implicit_queries = collect_implicit_candidate_queries(item)
        verify_results = collect_verify_results(item)
        if not verify_results:
            skipped["missing_verify"] += 1
            continue

        for verify_result in verify_results:
            row = build_grpo_row(
                item,
                verify_result,
                implicit_queries,
                min_confidence_by_label=thresholds,
            )
            if row is None:
                skipped["invalid_or_rejected_verify"] += 1
                continue
            source_id = item.get("source_id")
            candidates.append(
                Candidate(
                    row=row,
                    label=label,
                    source_id=str(source_id) if source_id is not None else None,
                    source_index=source_index,
                )
            )

    return candidates, skipped


def normalize_sample_fractions(weights: Dict[str, float]) -> Dict[str, float]:
    fractions: Dict[str, float] = {}
    for label in VALID_LABELS:
        value = float(weights.get(label, 0.0))
        fractions[label] = max(0.0, min(1.0, value))
    return fractions


def sample_candidates(
    candidates: List[Candidate],
    label_sample_weights: Optional[Dict[str, float]] = None,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Candidate], Dict[str, int]]:
    if not candidates:
        return [], {}

    fractions = normalize_sample_fractions(label_sample_weights or LABEL_SAMPLE_WEIGHTS)
    by_label: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.label].append(candidate)

    rng = random.Random(seed)
    for label in VALID_LABELS:
        rng.shuffle(by_label[label])

    quotas = {
        label: int(round(len(by_label[label]) * fractions[label]))
        for label in VALID_LABELS
    }

    selected: List[Candidate] = []
    for label in VALID_LABELS:
        selected.extend(by_label[label][: quotas[label]])
    return selected, quotas


def group_candidates_by_source_id(candidates: List[Candidate]) -> List[Candidate]:
    indexed_candidates = list(enumerate(candidates))

    def sort_key(indexed_candidate: Tuple[int, Candidate]) -> Tuple[int, str, int, int]:
        candidate_index, candidate = indexed_candidate
        if candidate.source_id is None:
            return (1, "", candidate.source_index, candidate_index)
        return (0, candidate.source_id, candidate.source_index, candidate_index)

    return [
        candidate
        for _, candidate in sorted(indexed_candidates, key=sort_key)
    ]


def shuffle_output_rows(
    rows: List[Dict[str, Any]],
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)
    return shuffled_rows


def _fraction(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample fraction must be a number.") from exc
    if not 0.0 <= fraction <= 1.0:
        raise argparse.ArgumentTypeError("sample fraction must be between 0 and 1.")
    return fraction


def _confidence_threshold(value: str) -> int:
    try:
        threshold = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confidence threshold must be an integer.") from exc
    if not 0 <= threshold <= 100:
        raise argparse.ArgumentTypeError("confidence threshold must be between 0 and 100.")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert synthesized Syn-UC samples to MS-SWIFT GRPO data.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--format", choices=("json", "jsonl"), default="jsonl")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--normal-fraction", type=_fraction, default=1.0)
    parser.add_argument("--explicit-fraction", type=_fraction, default=0.33)
    parser.add_argument("--implicit-fraction", type=_fraction, default=1.0)
    parser.add_argument("--min-normal-confidence", type=_confidence_threshold, default=95)
    parser.add_argument("--min-explicit-confidence", type=_confidence_threshold, default=100)
    parser.add_argument("--min-implicit-confidence", type=_confidence_threshold, default=90)
    parser.add_argument("--include-id", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used only for final row shuffling.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    items = read_jsonl(args.input)
    thresholds = {
        "Normal Dialogue": args.min_normal_confidence,
        "Explicit Revision": args.min_explicit_confidence,
        "Implicit Conflict": args.min_implicit_confidence,
    }
    fractions = {
        "Normal Dialogue": args.normal_fraction,
        "Explicit Revision": args.explicit_fraction,
        "Implicit Conflict": args.implicit_fraction,
    }
    candidates, skipped = build_all_candidates(
        items,
        min_confidence_by_label=thresholds,
    )
    selected, quotas = sample_candidates(
        candidates,
        label_sample_weights=fractions,
        seed=args.seed,
    )
    selected = group_candidates_by_source_id(selected)

    rows = [
        to_ms_swift_row(candidate.row, include_id=args.include_id)
        for candidate in selected
    ]
    if args.shuffle:
        rows = shuffle_output_rows(rows, seed=args.shuffle_seed)
        print(f"[Info] Shuffled output rows with seed {args.shuffle_seed}.")
    write_rows(args.output, rows, args.format)

    print(
        "[Done] "
        f"input_items={len(items)} candidates={len(candidates)} rows={len(rows)} "
        f"candidate_labels={dict(Counter(c.label for c in candidates))} "
        f"row_labels={dict(Counter(row['ground_truth'] for row in rows))} "
        f"quotas={quotas} skipped={dict(skipped)}"
    )
    print(f"[Written] {args.output}")


if __name__ == "__main__":
    main()
