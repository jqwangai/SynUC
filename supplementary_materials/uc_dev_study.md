# UC-Dev Hyperparameter Study for SynUC

## UC-Dev construction

UC-Dev contains 60 human-constructed instances grounded in naturally occurring dialogue histories, with a balanced class distribution: 20 Normal Dialogue (ND), 20 Explicit Revision (ER), and 20 Implicit Conflict (IC). It contains 30 English and 30 Chinese instances. Twenty task scenarios are each paired with one ND, one ER, and one IC continuation, holding dialogue history constant within each counterfactual triplet. UC-Dev follows the same annotation and quality-control protocol as UC-Bench. It uses previously unused source dialogues and has no source-dialogue overlap with UC-Data or UC-Bench. Source conversations were separated before any rewriting or counterfactual continuation construction.

## Protocol

Each row below is computed from 60 per-instance predictions. We report macro-precision (P), macro-recall (R), macro-F1 (F1), and IC recall (R(IC)), all as percentages. Because each class has only 20 instances, class recall changes in increments of five percentage points.

## Results

### Confidence threshold and memory configuration

*Table 1. Effect of confidence threshold $q$. Bold indicates the selected default.*

| $q$ | P | R | F1 | R(IC) |
|---:|---:|---:|---:|---:|
| 70 | 78.33 | 78.33 | 78.11 | 90.00 |
| 75 | 80.16 | 80.00 | 79.97 | 85.00 |
| **80** | 81.87 | 81.67 | 81.63 | 85.00 |
| 85 | 80.20 | 80.00 | 79.97 | 80.00 |
| 90 | 79.44 | 78.33 | 78.35 | 70.00 |

$q=80$ achieves the highest development-set macro-F1 and is therefore selected.

*Table 2. Effect of $K$. In the implementation, memory capacity and update trigger are the same parameter and are not treated as independent hyperparameters.*

| $K$ | P | R | F1 | R(IC) |
|---:|---:|---:|---:|---:|
| 1 | 75.13 | 75.00 | 75.03 | 75.00 |
| **3** | 81.87 | 81.67 | 81.63 | 85.00 |
| 5 | 80.38 | 80.00 | 80.10 | 80.00 |

$K=3$ achieves the highest development-set macro-F1 and is therefore selected.

### Sequential reward-weight selection

*Table 3. Selection of format-reward weight $\alpha$ with $\beta=0.4$.*

| $\alpha$ | P | R | F1 | R(IC) |
|---:|---:|---:|---:|---:|
| 0.1 | 78.69 | 78.33 | 78.38 | 80.00 |
| **0.2** | 81.87 | 81.67 | 81.63 | 85.00 |
| 0.3 | 80.13 | 80.00 | 80.03 | 80.00 |

*Table 4. Selection of attribution-reward weight $\beta$ with selected $\alpha=0.2$.*

| $\beta$ | P | R | F1 | R(IC) |
|---:|---:|---:|---:|---:|
| 0.2 | 78.63 | 78.33 | 78.39 | 80.00 |
| **0.4** | 81.87 | 81.67 | 81.63 | 85.00 |
| 0.6 | 80.13 | 80.00 | 80.03 | 80.00 |

In the sequential study, $\alpha=0.2$ achieves the highest development-set macro-F1 with $\beta=0.4$ and is selected first. With $\alpha=0.2$ fixed, $\beta=0.4$ achieves the highest development-set macro-F1 and is selected.

### Context retention and retrieval

*Table 5. Effect of assistant-response retention.*

| Setting | P | R | F1 | R(IC) |
|:---|---:|---:|---:|---:|
| Full omission | 80.13 | 80.00 | 80.03 | 80.00 |
| **Partial omission** | 81.87 | 81.67 | 81.63 | 85.00 |
| Full retention | 78.69 | 78.33 | 78.38 | 80.00 |

Partial omission achieves the highest development-set macro-F1 and is therefore selected.

*Table 6. Effect of schema retrieval strategy.*

| Strategy | P | R | F1 | R(IC) |
|:---|---:|---:|---:|---:|
| Lexical retrieval | 78.63 | 78.33 | 78.39 | 80.00 |
| Dense retrieval | 76.79 | 76.67 | 76.70 | 80.00 |
| **LLM-guided retrieval** | 81.87 | 81.67 | 81.63 | 85.00 |

LLM-guided retrieval achieves the highest development-set macro-F1 and is therefore selected.

### Generator LLM

*Table 7. Effect of the generator LLM.*

| Generator | P | R | F1 | R(IC) |
|:---|---:|---:|---:|---:|
| **DeepSeek-V4-Flash** | 81.87 | 81.67 | 81.63 | 85.00 |
| Doubao-Seed-2.0-Lite | 74.31 | 73.33 | 73.32 | 65.00 |
| MiMo-V2.5 | 71.33 | 70.00 | 69.66 | 55.00 |
| Qwen3.7 Plus | 75.30 | 73.33 | 73.25 | 60.00 |

DeepSeek-V4-Flash achieves the highest macro-F1 on UC-Dev and is therefore retained as the generator for the final configuration.

## Selected configuration

Across these one-factor-at-a-time comparisons, the selected configuration is $q=80$, $K=3$, $\alpha=0.2$, $\beta=0.4$, partial omission, LLM-guided retrieval, and DeepSeek-V4-Flash.
