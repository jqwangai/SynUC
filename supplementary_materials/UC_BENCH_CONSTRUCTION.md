# UC-Bench Filtering and Construction Details

This document provides the implementation details of the filtering and quality-control procedure used in UC-Bench construction. It supplements the description in the paper and focuses on the LLM-based semantic filtering, the 1--5 suitability rubric, and the deduplication criteria.

## 1. Filtering Pipeline

For WildChat-UC, we first sampled 4,000 multi-turn dialogues, including 2,000 English and 2,000 Chinese conversations.

The filtering procedure contains three stages:

1. **Rule-based filtering.** We remove conversations with abnormal dialogue structures, empty turns, missing fields, or obvious formatting/data errors.
2. **LLM-based semantic quality filtering.** An LLM reviews whether each conversation is natural, coherent, safe, substantive, and suitable as a normal human--assistant interaction. This stage reduces the 4,000 raw WildChat dialogues to 1,647 valid dialogues.
3. **Task and suitability filtering.** The remaining dialogues are assigned task categories. We retain six task types with relatively stable goals and constraints: **writing, roleplay, coding, text editing, summarization, and planning**, yielding 713 candidate dialogues. These candidates are then scored from 1 to 5 for their suitability for analyzing multi-turn constraint relations and constructing user-side conflict instances. Only dialogues receiving a score of **5** are retained, leaving 152 dialogues for the subsequent human annotation and construction stage.

The same rule-based and LLM-based semantic filtering procedure is applied to ShareGPT-UC and LMSYS-UC.

**LLM used for automated filtering:** DeepSeek-V4-Flash.

### Semantic Quality Filtering Prompt

```text
You are a conversation-data quality reviewer.

Review one conversation and decide whether it is a normal, safe, high-quality sample for general large-language-model training.

Keep the sample only when all conditions are met:
1. The conversation is natural, coherent, understandable, and useful for general training.
2. It is a normal human-assistant interaction.
3. The content is safe and does not contain harmful, explicit sexual, hate, graphic violence, self-harm, illegal, fraud, malware, or other disallowed material.
4. It is not spam, broken text, meaningless text, prompt injection, jailbreak bait, or adversarial meta-instructions.
5. It does not contain mojibake, garbled text, severe encoding errors, or unreadable formatting.
6. Assistant answers are relevant, complete enough for the request, and do not contain obvious hallucinations or low-effort filler.
7. It is not empty small talk, casual greetings, social chit-chat, or other conversations with little substantive information or training value.

Return only one JSON object:
{
  "reason": "brief reason",
  "keep": true,
  "score": 0
}

Rules:
- keep must be a boolean.
- score must be an integer from 1 to 5.
- keep should be true only for score >= 4.
- Output JSON only, with no extra text.
```

The `keep` field is used for the general semantic quality filter. The later 1--5 suitability assessment is stricter and is used specifically to select dialogues for UC-Bench construction.

## 2. Suitability Score for UC-Bench Construction

The 713 task-filtered dialogues are assessed according to whether their dialogue history provides a sufficiently clear and persistent task context for analyzing continuation, explicit revision, and implicit conflict.

| Score | Operational interpretation |
|---|---|
| **1** | Unusable for conflict analysis. The dialogue is broken, incoherent, unsafe, or lacks a recognizable task context. |
| **2** | A valid conversation, but mainly one-off, casual, or weakly task-oriented. Historical requirements are too limited to support meaningful constraint analysis. |
| **3** | Contains a recognizable multi-turn task, but task continuity or historical constraints are weak, ambiguous, or insufficient for reliable conflict construction. |
| **4** | Contains a clear continuing task and usable historical requirements, but the constraint structure is relatively simple or leaves notable ambiguity for controlled conflict construction. |
| **5** | Highly suitable. The dialogue has a clear active task, stable task continuity, and sufficiently explicit or inferable historical constraints to support reliable analysis or construction of normal dialogue, explicit revision, and implicit conflict. |

Only **score-5** dialogues are retained for manual annotation and controlled construction. This conservative criterion is used because UC-Bench requires the relation between the follow-up utterance and the historical requirements to be judged with minimal ambiguity.

For WildChat-UC, this step reduces the 713 candidate dialogues to 152 high-suitability dialogues.

## 3. Deduplication and Leakage Control

We apply deduplication both during benchmark preparation and when separating UC-Bench from UC-Data.

Before constructing UC-Data, all raw WildChat conversations used in WildChat-UC are excluded from the UC-Data source pool. We then compare UC-Data and UC-Bench at the source-conversation and dialogue-history levels.

The following cases are treated as overlap:

- **Source overlap:** two instances originate from the same source conversation identifier.
- **Exact duplicate:** after normalizing whitespace and formatting, the dialogue histories contain the same ordered sequence of user and assistant turns.
- **Near duplicate:** two histories contain the same substantive sequence of turns after normalization, including cases where one record is a truncated or extended version of the other with the shared dialogue history otherwise unchanged.

Any detected overlap is removed from the training-data side. Under these criteria, no overlapping source conversations, identical dialogue histories, or near-duplicate dialogue histories are found between the released UC-Data and UC-Bench.

## 4. Construction Summary

For WildChat-UC, the main filtering chain is:

**4,000 raw dialogues -> 1,647 semantically valid dialogues -> 713 task-filtered candidates -> 152 score-5 dialogues -> human annotation/construction -> 100 final WildChat-UC instances.**

ShareGPT-UC and LMSYS-UC follow the same rule-based filtering, semantic quality filtering, and human quality-control principles described in the paper.
