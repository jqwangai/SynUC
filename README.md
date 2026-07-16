# SynUC: Constraint-Guided Synthesis for User-Side Conflict Detection

> 🧩 Detecting implicit conflicts in follow-up user utterances before an LLM
> responds.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Resources](https://img.shields.io/badge/Resources-Coming_Soon-orange)](#-open-source-resources)

In human–LLM dialogue, a follow-up user utterance may implicitly conflict with
an earlier intent. If the conflict goes unnoticed, the LLM can misinterpret the
user's needs and produce an inappropriate response. A reliable dialogue system
should detect these **user-side conflicts** before responding and ask for
clarification when needed. While prior work has largely studied conflicts on the
LLM side, user-side conflicts remain underexplored.

**SynUC** is a constraint-guided data-synthesis method for user-side conflict
detection. It represents dialogue conflicts in a constraint space and uses the
**SPEAKING** framework to guide traceable transformations between historical and
current user constraints. This explicit modeling makes it possible to follow how
a conflict evolves and to generate implicit-conflict samples with reliable
labels.

This repository provides the SynUC synthesis and training-data conversion
pipeline. Given a dialogue history, the pipeline extracts active constraints,
generates a follow-up query for an enabled target category, and retains the
sample only after an independent verification step.

## 🌟 Research highlights

- **UC-Bench** is a human-annotated benchmark designed to evaluate user-side
  conflict detection, including difficult implicit incompatibilities grounded in
  dialogue history. Experiments show that existing LLMs struggle with this task,
  especially on implicit conflicts.
- Applying SynUC to **WildChat** produces **UC-Data**, a user-side conflict
  training set containing **2,487 samples**.
- On UC-Bench, **Qwen3.5-4B** trained on UC-Data outperforms larger
  general-purpose LLMs, including **Claude Opus 4.8**, as well as models trained
  with existing synthesis methods.

SynUC synthesizes and verifies three categories:

- ⚡ **Implicit Conflict**: the new query implicitly introduces a constraint that
  cannot be satisfied together with an active historical constraint.
- 🔄 **Explicit Revision**: the new query explicitly replaces or revises an active
  historical constraint.
- ✅ **Normal Dialogue**: the new query remains compatible with the active
  historical constraints.

## 📦 Open-source resources

The dataset, benchmark, and trained model will be released here. This table is
the central entry point for all open-source project artifacts.

| Resource | Description | Access |
| --- | --- | --- |
| 🗃️ **UC-Data** | A 2,487-sample user-side conflict training set synthesized from WildChat. | *Coming soon* |
| 📊 **UC-Bench** | A human-annotated benchmark for user-side conflict detection. | *Coming soon* |
| 🧠 **Model Weights** | Qwen3.5-4B trained on UC-Data for user-side conflict detection. | *Coming soon* |

> 🔔 Watch this repository for release updates. Download links and usage
> instructions will be added to this section once the resources are public.

## 🗂️ Repository structure

```text
Syn_UC_github/
├── synthesize_dataset.py        # Command-line entry point
├── convert_to_sft.py            # SFT data conversion entry point
├── convert_to_grpo.py           # GRPO data conversion entry point
├── requirements.txt             # Python dependencies
├── examples/
│   └── synthetic_dataset.jsonl  # Complete conversion example
├── tests/
│   └── test_conversions.py      # Offline conversion tests
├── resources/
│   └── memory_patterns.yaml     # Immutable synthesis memory
├── data/
│   ├── input/                   # Source dialogues
│   ├── output/                  # Synthesized samples and token statistics
│   ├── cache/                   # Task and constraint-extraction caches
│   ├── runtime/                 # Mutable memory and trajectory buffer
│   └── training/                # Converted SFT and GRPO data
└── syn_uc/
    ├── config.py                # Paths, model settings, and synthesis switches
    ├── pipeline.py              # End-to-end parallel synthesis pipeline
    ├── prompts.py               # LLM prompts
    ├── dialogue_utils.py        # Dialogue normalization and prompt helpers
    ├── memory_ops.py            # Memory loading and pattern retrieval
    ├── trajectory_buffer.py     # Trajectory accumulation and promotion
    ├── io_cache.py              # JSONL I/O and persistent caches
    ├── token_usage.py           # Thread-safe token accounting
    ├── conversion_utils.py      # Shared conversion and schema utilities
    ├── sft_conversion.py        # SFT message construction
    └── grpo_conversion.py       # GRPO sampling and row construction
```

## 🛠️ Installation

Python 3.10 or later is recommended.

```bash
cd Syn_UC_github
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## 📥 Input format

Place the source data at `data/input/dialogues.jsonl`, or override the path with
`SYN_UC_INPUT_PATH`. Each JSONL record must include a `language` field and use one
of the following schemas:

```json
{
  "id": "dialogue_id",
  "language": "en",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

```json
{
  "conversation_id": "dialogue_id",
  "language": "en",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Only dialogues containing more than one user turn are processed. A minimal
record is provided at `data/input/dialogues.example.jsonl`.

## ⚙️ Configuration

Set an API key before running. The client accepts any OpenAI-compatible endpoint.

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export SYN_UC_MODEL="deepseek-v4-flash"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key"
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
$env:SYN_UC_MODEL = "deepseek-v4-flash"
```

The main synthesis switches are defined in `syn_uc/config.py`:

| Setting | Purpose |
| --- | --- |
| `ENABLE_IMPLICIT_CONFLICT` | Generate implicit-conflict samples. |
| `ENABLE_EXPLICIT_REVISION` | Generate explicit-revision samples. |
| `ENABLE_NORMAL_DIALOGUE` | Retain compatible follow-up queries. |
| `ENABLE_TASK_FILTER` | Filter unsuitable source dialogues. |
| `ENABLE_TRAJECTORY_BUFFER` | Accumulate reusable implicit-conflict trajectories. |
| `SAMPLE_LIMIT` | Limit eligible dialogues; use `None` for all records. |
| `PARALLEL_WORKERS` | Number of dialogue-level worker threads. |

Paths can be overridden without editing the code:

| Environment variable | Default |
| --- | --- |
| `SYN_UC_INPUT_PATH` | `data/input/dialogues.jsonl` |
| `SYN_UC_OUTPUT_PATH` | `data/output/synthetic_dataset.jsonl` |
| `SYN_UC_TOKEN_USAGE_PATH` | `data/output/token_usage.jsonl` |
| `SYN_UC_TASK_FILTER_CACHE_PATH` | `data/cache/task_filter.jsonl` |
| `SYN_UC_CONSTRAINT_CACHE_PATH` | `data/cache/active_constraints.jsonl` |

Relative override paths are resolved from the repository root.

## 🚀 Running synthesis

```bash
python synthesize_dataset.py
```

Accepted samples are appended to `data/output/synthetic_dataset.jsonl`. Token
statistics are appended to `data/output/token_usage.jsonl`. Task-filter and
constraint-extraction results are cached, so interrupted runs can reuse completed
LLM calls. Remove or redirect an existing output file before starting an
independent experiment if appending is not desired.

## 📤 Output records

Every accepted record contains:

- `sample_id`: identifier derived from the source dialogue and target category;
- `source_id`: identifier of the source dialogue;
- `language`: copied from the input record;
- `label`: `Implicit Conflict`, `Explicit Revision`, or `Normal Dialogue`;
- `dialogue_history`: the source dialogue used for synthesis;
- `current_query` or `candidate_queries`: verified generated query data;
- `meta`: intermediate LLM outputs, verification details, and token usage.

Generation failures are reported in the terminal and are not written as accepted
training samples. Because model sampling and concurrent completion order can vary,
exact output ordering is not guaranteed across runs.

## 🧪 Training data conversion

Both converters read the synthesized JSONL schema described above. By default,
they use `data/output/synthetic_dataset.jsonl`, so the synthesis output can be
converted without moving or renaming it.

### 🎯 SFT data

Convert accepted samples into user/assistant classification messages:

```bash
python convert_to_sft.py
```

The default output is a JSON array at `data/training/sft.json`. Each row contains
one user prompt and one assistant response. The assistant response follows the
`<think>...</think><<label>>` format used for supervised fine-tuning.

Useful options include:

```bash
python convert_to_sft.py \
  --input path/to/synthetic_dataset.jsonl \
  --output data/training/sft.jsonl \
  --format jsonl \
  --sample-limit all \
  --include-metadata
```

The converter excludes rejected records, mismatched verification labels, missing
queries, and candidates below the label-specific confidence threshold. By
default, the minimum confidence is 95 for Normal Dialogue, 100 for Explicit
Revision, and 90 for Implicit Conflict. These thresholds can be overridden from
the command line.

### 🏆 GRPO data

Convert accepted samples into MS-SWIFT-compatible GRPO rows:

```bash
python convert_to_grpo.py
```

The default output is `data/training/grpo.jsonl`. Each row contains:

- `messages`: the classification prompt;
- `ground_truth`: the verified class label;
- `ground_truth_history_constraint`: the conflicting historical constraint for
  conflict or revision samples;
- `ground_truth_new_constraint`: the conflicting new constraint for conflict or
  revision samples.

The converter performs deterministic per-label sampling. The default retained
fractions are 1.0 for Normal Dialogue, 0.33 for Explicit Revision, and 1.0 for
Implicit Conflict. Example:

```bash
python convert_to_grpo.py \
  --input path/to/synthetic_dataset.jsonl \
  --output data/training/grpo.jsonl \
  --normal-fraction 1.0 \
  --explicit-fraction 0.33 \
  --implicit-fraction 1.0 \
  --seed 42 \
  --shuffle
```

Use `--include-id` to retain traceable sample IDs. Run either converter with
`--help` for the complete option list.

### 💡 Conversion example

The repository includes a complete synthesized record that can be converted
without calling an external model:

```bash
python convert_to_sft.py \
  --input examples/synthetic_dataset.jsonl \
  --output data/training/example_sft.json

python convert_to_grpo.py \
  --input examples/synthetic_dataset.jsonl \
  --output data/training/example_grpo.jsonl \
  --explicit-fraction 1.0
```

Conversion outputs are overwritten on each invocation; synthesis outputs are
appended by the synthesis pipeline.

Run the offline conversion tests with:

```bash
python -m unittest discover -s tests
```

