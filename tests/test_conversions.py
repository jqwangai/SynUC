"""Unit tests for SFT and GRPO data conversion."""

import unittest

from syn_uc import grpo_conversion, sft_conversion


def verification(
    label: str,
    confidence: int,
    candidate_id: str = "0",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "label": label,
        "confidence": confidence,
        "task_judge": "current_subtask",
        "history": [
            {"source": "explicit", "dimension": "language", "value": "English"}
        ],
        "new_explicit_constraints": [
            {"source": "explicit", "dimension": "language", "value": "Chinese"}
        ],
        "new_implicit_constraints": [],
        "channel": "H_E_N_E",
        "reasoning": "The new language constraint replaces the historical one.",
    }


def base_item(sample_id: str, label: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_id": sample_id,
        "label": label,
        "language": "en",
        "dialogue_history": [
            {"role": "user", "content": "Write a short announcement in English."},
            {"role": "assistant", "content": "Here is a short announcement."},
        ],
        "meta": {"llm_outputs": {}},
    }


def make_normal_item() -> dict:
    item = base_item("normal_sample", "Normal Dialogue")
    item["current_query"] = "Add the event date."
    item["meta"]["llm_outputs"]["verify"] = {
        "results": [verification("Normal Dialogue", 95)]
    }
    return item


def make_explicit_item() -> dict:
    item = base_item("explicit_sample", "Explicit Revision")
    item["current_query"] = "Rewrite it in Chinese."
    item["meta"]["llm_outputs"]["verify"] = {
        "results": [verification("Explicit Revision", 100)]
    }
    item["meta"]["llm_outputs"]["generate_explicit_revision"] = {
        "generation": {
            "new_constraints": [
                {
                    "candidate_id": "0",
                    "historical_constraint": {
                        "source": "explicit",
                        "dimension": "language",
                        "value": "English",
                    },
                    "new_constraint": {
                        "source": "explicit",
                        "dimension": "language",
                        "value": "Chinese",
                    },
                }
            ]
        }
    }
    return item


def make_implicit_item() -> dict:
    item = base_item("implicit_sample", "Implicit Conflict")
    result = verification("Implicit Conflict", 90, candidate_id="7")
    result["new_explicit_constraints"] = []
    result["new_implicit_constraints"] = [
        {"source": "implicit", "dimension": "language", "value": "Chinese"}
    ]
    result["channel"] = "H_E_N_I"
    item["candidate_queries"] = [
        {
            "candidate_id": "7",
            "current_query": "Prepare it for publication in a Chinese newspaper.",
            "verification": result,
        }
    ]
    item["meta"]["llm_outputs"]["generate_implicit"] = {
        "new_constraints_generation": [
            {
                "candidate_id": "7",
                "historical_constraint": {
                    "source": "explicit",
                    "dimension": "language",
                    "value": "English",
                },
                "new_constraint": {
                    "source": "implicit",
                    "dimension": "language",
                    "value": "Chinese",
                },
            }
        ]
    }
    return item


class ConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = [make_normal_item(), make_explicit_item(), make_implicit_item()]

    def test_sft_conversion_covers_all_labels(self) -> None:
        records = sft_conversion.convert_to_train_data(
            self.items,
            include_metadata=True,
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(
            {record["label"] for record in records},
            {"Normal Dialogue", "Explicit Revision", "Implicit Conflict"},
        )
        self.assertTrue(all(len(record["messages"]) == 2 for record in records))

    def test_grpo_conversion_preserves_constraint_ground_truth(self) -> None:
        candidates, skipped = grpo_conversion.build_all_candidates(self.items)
        self.assertEqual(len(candidates), 3)
        self.assertFalse(skipped)

        selected, quotas = grpo_conversion.sample_candidates(
            candidates,
            label_sample_weights={label: 1.0 for label in grpo_conversion.VALID_LABELS},
            seed=42,
        )
        self.assertEqual(sum(quotas.values()), 3)
        rows = [grpo_conversion.to_ms_swift_row(item.row) for item in selected]
        conflict_rows = [row for row in rows if row["ground_truth"] != "Normal Dialogue"]
        self.assertEqual(len(conflict_rows), 2)
        self.assertTrue(
            all(row["ground_truth_history_constraint"]["value"] == "English" for row in conflict_rows)
        )
        self.assertTrue(
            all(row["ground_truth_new_constraint"]["value"] == "Chinese" for row in conflict_rows)
        )


if __name__ == "__main__":
    unittest.main()
