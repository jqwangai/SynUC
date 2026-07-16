"""Runtime configuration for Syn-UC dataset synthesis."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESOURCE_DIR = PROJECT_ROOT / "resources"


def _project_path(environment_variable: str, default: Path) -> str:
    """Resolve an optional environment path relative to the project root."""
    configured = os.getenv(environment_variable)
    path = Path(configured) if configured else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


INPUT_PATH = _project_path(
    "SYN_UC_INPUT_PATH",
    DATA_DIR / "input" / "dialogues.jsonl",
)
OUTPUT_PATH = _project_path(
    "SYN_UC_OUTPUT_PATH",
    DATA_DIR / "output" / "synthetic_dataset.jsonl",
)
TOKEN_USAGE_PATH = _project_path(
    "SYN_UC_TOKEN_USAGE_PATH",
    DATA_DIR / "output" / "token_usage.jsonl",
)
TASK_FILTER_CACHE_PATH = _project_path(
    "SYN_UC_TASK_FILTER_CACHE_PATH",
    DATA_DIR / "cache" / "task_filter.jsonl",
)
EXTRACT_FEATURES_CACHE_PATH = _project_path(
    "SYN_UC_CONSTRAINT_CACHE_PATH",
    DATA_DIR / "cache" / "active_constraints.jsonl",
)

# The source memory is never modified. Each run copies it to the runtime area.
MEMORY_TEMPLATE_PATH = RESOURCE_DIR / "memory_patterns.yaml"
MEMORY_PATH = DATA_DIR / "runtime" / "memory_patterns.yaml"

ENABLE_TRAJECTORY_BUFFER = True
TRAJECTORY_BUFFER_PATH = DATA_DIR / "runtime" / "trajectory_buffer.yaml"
TRAJECTORY_BUFFER_THRESHOLD = 3
TRAJECTORY_BUFFER_POSITIVE_MIN_CONFIDENCE = 95
TRAJECTORY_BUFFER_MAX_POSITIVE_TRAJECTORIES = 3
TRAJECTORY_BUFFER_MAX_NEGATIVE_TRAJECTORIES = 3

MODEL = os.getenv("SYN_UC_MODEL", "deepseek-v4-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
ENABLE_DEEP_THINKING = False
REASONING_EFFORT = "high"

MEMORY_RETRIEVAL_TOP_K_PER_CONSTRAINT = 1
MEMORY_SELECTED_PAIR_LIMIT = 1
MEMORY_RETRIEVAL_MIN_SCORE = 0.45
MEMORY_DEDUP_BY_DIMENSION = True

# None processes every eligible dialogue; use a small integer for a smoke test.
SAMPLE_LIMIT = 2000

MAX_IMPLICIT_PER_DIALOGUE = 1
MAX_EXPLICIT_PER_DIALOGUE = 1
ENABLE_TASK_FILTER = True
ENABLE_IMPLICIT_CONFLICT = False
ENABLE_EXPLICIT_REVISION = True
ENABLE_NORMAL_DIALOGUE = False
NORMAL_DIALOGUE_PROBABILITY_FOR_SELECTED_TASK = 0.0

TRUNCATE_ASSISTANT_IN_PROMPTS = True
ASSISTANT_TRUNCATE_HEAD_CHARS = 40
ASSISTANT_TRUNCATE_TAIL_CHARS = 40
ASSISTANT_TRUNCATE_MIN_CHARS = 120
SLEEP_SECONDS = 0.0
PARALLEL_WORKERS = 10
