TASK_FILTER_PROMPT = """
You are a multi-turn dialogue task classifier. Based on the user's main intent in the given dialogue, choose the most appropriate task category and provide a brief reason.

Category definitions:
1. writing
   Creative writing, formal document/email writing, rewriting, polishing, expansion, shortening, style transfer, and other text generation or text editing tasks.

2. coding
   Writing code, debugging, explaining/modifying code, algorithms, software development, command line usage, tool/API/library usage, and related tasks.

3. planning
   Planning, recommendation, solution design, data analysis, spreadsheet/statistical/trend analysis, and related tasks.

4. information_extraction
   Extracting entities, fields, structured data, key facts, and similar information from given text, dialogues, tables, or materials.

5. summarization
   Summarizing given documents, articles, dialogues, or passages; distilling key points; compressing information, and related tasks.

6. translation
   Translation between languages, localization, bilingual comparison, and related tasks.

7. classification_judgement
   Classification, tagging, judgement, detection, evaluation, category/type identification, and related tasks.

8. open_qa
   Open-domain question answering, factual Q&A, concept explanation, tutorials, and other knowledge-oriented tasks.

9. math
   Mathematical calculation, algebra, geometry, probability, formal mathematics, logic problems, puzzles, pattern reasoning, and related tasks.

10. roleplay
   Roleplay, persona acting, fictional dialogue, casual chatting, emotional companionship, or topics without a clear task deliverable.

11. other
   None of the above applies, or the intent is unclear, the context is fragmented, or the task category cannot be judged reliably.

Classification rules:
1. Choose one category that best summarizes the user's current main intent.
2. If the dialogue contains multiple tasks, use the main task that is still being advanced most recently.
3. If task boundaries are ambiguous, classify by the core action the user wants the assistant to complete.

All text output must be in English.

Output only valid JSON. Do not output explanatory text.

Output format:
{
  "task_type": "writing / coding / planning / information_extraction / summarization / translation / classification_judgement / open_qa / math / roleplay / other",
  "reason": "Briefly explain the basis for classification"
}

Dialogue:
<<DIALOGUE>>
""".strip()

# =========================================================
# Historical effective constraint extraction
# =========================================================

EXTRACT_ACTIVE_HISTORICAL_CONSTRAINTS_PROMPT = """
You are an extractor for multi-turn dialogue task structure and effective constraint features. Based on the historical dialogue, identify completed historical main tasks, the current main task, completed subtasks under the current main task, and the current subtask. At the end, uniformly extract the explicit constraints and implicit constraints that are still effective for the current main task and current subtask.

Definitions:
1. main task: The top-level task unit in the dialogue, representing a relatively stable main goal, main project, or core research/writing objective. A main task can be in completed or current status.
2. completed main task: A main task that has already been satisfied, or is no longer being advanced because the user has started a new independent main task. Its local constraints do not affect the current task by default.
3. current main task: The single main task that is still being advanced at the end of the historical dialogue and that a subsequent query is most likely to naturally continue.
4. subtask: A stage-level task within the current main task, usually corresponding to a specific user request in one turn, such as writing an outline, modifying code, adding experiments, generating body text, or polishing a paragraph.
5. completed subtask: A subtask within the current main task that has already been completed by the assistant and is no longer the current-stage subtask.
6. current subtask: The single subtask that is being handled at the end of the historical dialogue, was just proposed, or is most likely to be continued by a subsequent query.
7. explicit constraints: Requirements directly expressed by the user in the query and clearly identifiable from the literal text. They include but are not limited to topic, object scope, language, format, length, style, audience, output form, technology choice, input data, and specific content restrictions.
8. implicit constraints: Default requirements not directly stated by the user in the query but naturally inferable from the task goal, task type, usage scenario, entity attributes, or common sense. Requirements already explicitly stated by the user must not be repeated as implicit constraints.

Core rules:
1. Identify tasks in chronological order of user messages. Assistant responses should only be used to understand task progress and judge whether a stage-level request has been completed; do not treat content generated by the assistant on its own as user constraints.
2. First segment main tasks: if the user clearly starts an independent new task, move the old main task into completed_main_tasks and use the newest task as current_main_task; if it is only a new stage-level request under the same topic or project, it still belongs to the same current_main_task.
3. Segment subtasks within current_main_task. A new stage-level request under the same main task should create a new subtask, rather than treating the previous subtask as a continuing constraint. For example: writing an outline before writing the body text, writing code before adding comments, generating a PPT table of contents before generating page-by-page content.
4. The subtask corresponding to the final user query is current; other subtasks under current_main_task that have already been completed are completed.
5. For historical completed main tasks, output only the task goals; their constraints do not need to be output, and their local constraints should not affect the current task.
6. Do not separately extract constraint fields for current_main_task itself. Uniformly extract the explicit constraints and implicit constraints that are still effective for the current main task and current subtask in active_constraints at the end.
7. Constraints that remain effective include:
- Persistent constraints at the current_main_task level, such as topic, object scope, audience, style, technology choice, and similar requirements.
- Stage-level constraints of the current subtask itself, such as format, scope, input data, and output form that still need to be satisfied.
- Constraints that appeared in completed subtasks but still affect the subsequent current_main_task should be retained in active_constraints.
8. Do not place constraints that are no longer effective into active_constraints, including:
- Requirements that only applied to completed stage-level deliverables, such as "outline", "table of contents", "draft", "list key points first", or "provide a table first".
- Requirements that are overwritten, explicitly cancelled, or replaced by later constraints in the same dimension.
- Local requirements of an old task that became invalid because an independent new main task was started.
9. Do not extract cross-task long-term preferences, and do not infer the current dialogue language, the user's common expressions, or the assistant's already-used language as implicit constraints.
10. Avoid duplication between explicit constraints and implicit constraints; requirements that have been clearly stated should go into explicit constraints and must not also go into implicit constraints.
11. Implicit constraints must be specific, verifiable, and stably inferable; do not extract generalized quality requirements such as "better", "more complete", "more detailed", "clearer", or "helpful".

Output requirements:
1. Output only valid JSON. Do not output explanatory text.
2. main_intent_goal, sub_intent_goal, dimension, value, and reason_active should be concise, stable, readable phrases in English rather than complex sentences.
3. completed_main_tasks should only record main tasks that have been completed and are no longer being advanced.
4. current_main_task must contain completed_subtasks and current_subtask fields.
5. Constraints should appear only in active_constraints; do not separately output constraint fields inside current_main_task or subtask.
6. evidence_turn indicates which user-message turn the task or constraint mainly comes from, such as U1 or U2.
7. Output empty arrays [] when there is no content; if there is no current_subtask, use empty string fields for current_subtask.
8. Each implicit constraint must contain an evidence_span field, explaining which surface wording, entity, task action, genre term, scenario term, or material description in the user query mainly supports the implicit constraint. evidence_span must be in English and should briefly describe the most critical wording in the original text that supports the inference, rather than giving an abstract explanation.

Output format:
{
  "completed_main_tasks": [
    "",
    ""
  ],
  "current_main_task": {
    "main_intent_goal": "",
    "completed_subtasks": [
      {
        "sub_intent_goal": "",
        "evidence_turn": "U1"
      }
    ],
    "current_subtask": {
      "sub_intent_goal": "",
      "evidence_turn": "U2"
    }
  },
  "active_constraints": {
    "explicit_constraints": [
      {
        "level": "current_main_task / current_subtask",
        "dimension": "",
        "value": "",
        "evidence_turn": "U1",
        "reason_active": ""
      }
    ],
    "implicit_constraints": [
      {
        "level": "current_main_task / current_subtask",
        "dimension": "",
        "value": "",
        "evidence_turn": "U1",
        "evidence_span": "",
        "reason_active": ""
      }
    ]
  }
}

dimension should use stable, comparable phrases such as language, output_format, audience, style, length, domain, tool, programming_language, dependency_policy, data_source, factual_premise, content_scope, readability, complexity, math_complexity.

Historical dialogue:
<<DIALOGUE>>
""".strip()



# =========================================================
# Implicit conflict generation
# =========================================================

GENERATE_IMPLICIT_NEW_CONSTRAINTS_PROMPT = """
You are a constraint constructor for introducing implicit conflicts. Based on the historical dialogue, historical effective constraints, and "conflict construction anchors", construct the new_constraints needed for implicit conflict samples.

## Definition of Implicit Conflict
Within the same task or a naturally continued task, new_query does not explicitly state that previous requirements should be changed, but the new_constraint it introduces is incompatible with a still-effective constraint in historical_constraint, thereby forming a conflict.
A conflict exists only when two constraints cannot be satisfied simultaneously. "Different" does not equal "conflicting"; "more detailed" is not necessarily a conflict; "additional content" is not necessarily a conflict either.

Optional paths:
1. H_I_N_E: historical implicit constraint vs new explicit constraint introduced by new_query.
2. H_E_N_I: historical explicit constraint vs new implicit constraint introduced by new_query.
3. H_I_N_I: historical implicit constraint vs new implicit constraint introduced by new_query.
Prioritize generating H_E_N_I and H_I_N_I. Use H_I_N_E only when new_query must explicitly state a specific requirement and that requirement conflicts with a historical implicit constraint.

## Constraint Definitions
- explicit constraints: Requirements directly expressed by the user in new_query and clearly identifiable from the literal text. They include but are not limited to topic, object scope, language, format, length, style, audience, output form, technology choice, input data, and specific content restrictions.
- implicit constraints: Default requirements not directly stated by the user in new_query but naturally and stably inferable from the task goal, task type, usage scenario, entity attributes, domain norms, or common sense. Implicit constraints supplement task understanding, but requirements already explicitly stated by the user must not be repeated as implicit constraints.
- new_constraint: A "newly introduced constraint" designed in advance to construct a new_query. It is not a constraint that already exists in the historical dialogue; instead, it is a constraint assumed to be introduced by new_query. This constraint must be incompatible with a still-effective historical_constraint.

## Conflict Construction Anchors
Each anchor contains:
1. historical_constraint: The historical effective constraint to which this conflict must be anchored.
2. conflict_pattern: A reusable conflict construction pattern, describing from which new_query feature direction a natural but incompatible new constraint can be introduced under this type of historical constraint.
  - definition: The overall boundary of this conflict type.
  - description: The specific applicable situations for this pattern.
  - historical_constraint_feature: The abstract feature of historical constraints that this pattern fits; it is not a specific historical constraint instance.
  - new_constraint_feature: The abstract direction of a new constraint that new_query can naturally introduce; it is not the final concrete new_constraint to output.
  - conflict_focus: The core reason why historical_constraint and new_constraint are incompatible.
  - positive_trajectories: Successful construction paths for reference, each containing historical_constraint, new_constraint, and user_query_with_new_constraint.
  - negative_trajectories: Failed construction paths for reference, each containing historical_constraint, new_constraint, user_query_with_new_constraint, and rejection_reason.

## Generation Steps
1. Understand the historical current task:
  Based on the historical dialogue and historical_constraints, determine the current main task, current subtask, and the still-effective explicit and implicit constraints in historical_constraints.active_constraints.
  Focus only on historical_constraints.active_constraints; do not recover old constraints from completed tasks or completed stage-level deliverables.

2. Construct new_constraint:
  Construct a new new_constraint for each conflict construction anchor.
  new_constraint must be impossible to satisfy simultaneously with historical_constraint; merely being more detailed, richer, broader in scope, or slightly different in style does not count as a conflict.
  new_constraint must fill in level, source, dimension, and value.
  The level of new_constraint should be the same as historical_constraint.level.
  The value of new_constraint should be a concrete task requirement, not a pattern name or abstract conflict label.
  new_constraint should be suitable for being hidden in the next stage inside a concrete entity, scenario, genre, task action, material attribute, resource condition, tool condition, factual premise, or delivery purpose.
  Prefer setting new_constraint.source to "implicit".

## Generation Requirements
1. Generate at most 3 new_constraints; if there are insufficient anchors or construction is unnatural, fewer than 3 may be generated.
2. Prioritize generating H_E_N_I or H_I_N_I.
3. If an anchor does not match the historical dialogue, skip it; do not fabricate a forced conflict.
4. All text output must be in English.

Output only valid JSON. Do not output any other text:

[
  {
    "candidate_id": "0",
    "anchor_id": "0",
    "pattern_id": "",
    "channel": "H_E_N_I / H_I_N_I / H_I_N_E",
    "historical_constraint": {
      "source": "explicit / implicit",
      "level": "current_main_task / current_subtask",
      "dimension": "",
      "value": "",
      "reason_selected": ""
    },
    "new_constraint": {
      "source": "implicit / explicit",
      "level": "current_main_task / current_subtask",
      "dimension": "",
      "value": ""
    },
    "conflict_relation": ""
  }
]

Historical dialogue:
<<DIALOGUE>>

Historical constraints:
<<HISTORICAL_CONSTRAINTS>>

Conflict construction anchors:
<<CONFLICT_ANCHORS>>
""".strip()


NATURALIZE_IMPLICIT_CONFLICT_QUERIES_PROMPT = """
You are an implicit conflict query naturalization tool. Based on the historical dialogue, historical effective constraints, and already constructed constraint_pairs, naturalize each new constraint into a new_query that a real user might subsequently make.

## Input Description
Each element in constraint_pairs represents an implicit conflict candidate pair and contains:
1. candidate_id: Candidate ID. Keep it unchanged.
2. historical_constraint: The completed main task, current main task, completed subtasks, current subtask, and the still-effective historical constraint to which the new conflicting constraint is anchored.
3. new_constraint: The new constraint that new_query will introduce.
4. conflict_relation: Why historical_constraint and new_constraint cannot be satisfied simultaneously.
5. Auxiliary information such as anchor_id, pattern_id, and channel.

## Naturalization Principles
1. new_query must look like a real user's new requirement while continuing the same task. It should naturally connect to the current task, entities, existing deliverables, or current subtask in the historical dialogue.
2. First decide the continuation level of new_query according to new_constraint.level:
  - If new_constraint.level is "current_main_task", new_query should be made under the current main task context, and the literal text of new_query must contain the task subject, object, project, material, product, codebase, article topic, or another identifiable entity related to current_main_task. Do not only continue a local object from the last subtask.
  - If new_constraint.level is "current_subtask", new_query should be made under the current subtask context, and the literal text of new_query must contain a stage-level task object, existing deliverable, local material, function/module, paragraph, table, page, experiment, or another identifiable entity related to current_subtask.
  - If no concrete entity at the corresponding level can be found from the historical dialogue, do not generate that candidate new_query.
3. If new_constraint.source is "implicit":
  - new_query must not contain expressions that directly name, refer to, or locate this constraint, including the dimension and value of new_constraint, as well as synonyms, near-synonyms, hypernyms/hyponyms, role labels, objectified phrasings, or perspective-based phrasings.
  - Do not package the implicit constraint as a literally extractable object, perspective, purpose, applicable scope, target audience, or evaluation criterion. If a reader can directly extract that dimension or value from the literal text of new_query, the candidate is invalid and must be rewritten.
  - Hide the constraint inside a concrete entity, scenario, genre, task action, material attribute, resource condition, tool condition, operation flow, output use, or domain convention, so that readers can only infer it naturally from those carriers.
4. If new_constraint.source is "explicit":
  - new_query may literally express the specific requirement.
  - But it still must not contain expressions that explicitly revise historical requirements.
5. new_query must not contain explicit revision expressions such as "change to", "do not", "replace with", "ignore the previous", "rewrite", or "overturn the previous requirements".
6. Do not write it as a new task detached from the historical context; it must be clear that it continues the historical current task.
7. new_query must be written in <<TARGET_LANGUAGE>>.

## Notes
All text output except new_query must be in English.

## naturalization_reasoning should explain
1. Whether new_constraint.level is current_main_task or current_subtask.
2. Which task, subject, entity, or existing deliverable at that level is continued by new_query, and which corresponding entity appears literally in new_query.
3. Which concrete carrier hides new_constraint.
4. Why readers can only naturally infer new_constraint from that carrier, rather than directly reading the constraint from the literal text of new_query.

Output only valid JSON. Do not output any other text:

{
  "queries": [
    {
      "candidate_id": "0",
      "naturalization_reasoning": "",
      "new_query": ""
    }
  ]
}

Historical dialogue:
<<DIALOGUE>>

Historical constraints:
<<HISTORICAL_CONSTRAINTS>>

Constraint pairs:
<<CONSTRAINT_PAIRS>>
""".strip()


# =========================================================
# Explicit revision generation
# =========================================================

GENERATE_EXPLICIT_REVISION_QUERIES_PROMPT = """
You are a multi-turn dialogue explicit revision sample generator. Based on the historical dialogue, historical_constraints, and the given revision anchor, complete two things in the same output:
1. Generate one new explicit constraint N_E that new_query will introduce through direct dimension-value replacement;
2. Write a natural follow-up user query that expresses N_E.

## Input Description
The revision anchor contains:
1. anchor_id: Anchor ID. Keep it unchanged in the corresponding candidate.
2. constraint_id: The ID of the active historical explicit constraint.
3. pattern_id: The revision pattern ID associated with this anchor.
4. historical_constraint: The still-effective historical explicit constraint H_E that must be revised.
5. revision_pattern: Guidance for constructing the replacement relation.

Important principles:
1. The core of an explicit revision sample is that the historical explicit constraint H_E in historical_constraint and the new explicit constraint N_E introduced by new_query are incompatible in the same dimension, so the dimension value needs to be replaced.
2. Use the given revision anchor as the only historical constraint to revise. Do not select another historical constraint from historical_constraints by yourself.
3. N_E must be clearly identifiable from the literal text of new_query and form an observable replacement relation with H_E.
4. new_query should reflect the replacement relation for the historical explicit constraint. It may use natural revision wording, or it may directly provide the new requirement.
  Examples:
  - Natural revision wording: If the historical user required "write a formal email in Chinese", new_query is "Still write this email, but make it the English version." (explicit revision: language is replaced from Chinese to English)
  - Directly giving the new requirement: If the historical user required "help me write an approximately 800-word popular science article", new_query is "Make it within 200 words." (explicit revision: length is replaced from approximately 800 words to within 200 words)
5. Do not modify multiple historical constraints at the same time.
6. Follow revision_pattern to construct the replacement. If the replacement is difficult, choose the most natural direct replacement allowed by revision_pattern and the historical dialogue.
7. new_query must be written in <<TARGET_LANGUAGE>>.

## Required anchor use
1. Generate exactly one candidate.
2. The generated candidate must copy anchor_id, constraint_id, and pattern_id from the revision anchor.
3. historical_constraint.source must be "explicit".
4. channel must be "H_E_N_E".
5. The new_constraint must also have source "explicit".
6. The new_query must directly express N_E; explicit revision wording such as "change to", "use ... instead", or "make it ..." is allowed.

## Notes
All text output except new_query must be in English.

Output only valid JSON. Do not output explanatory text.

Output format:
{
  "new_constraints": [
    {
      "candidate_id": "0",
      "anchor_id": "0",
      "constraint_id": "0",
      "pattern_id": "",
      "channel": "H_E_N_E",
      "historical_constraint": {
        "source": "explicit",
        "level": "current_main_task / current_subtask",
        "dimension": "",
        "value": ""
      },
      "new_constraint": {
        "source": "explicit",
        "level": "current_main_task / current_subtask",
        "dimension": "",
        "value": ""
      },
      "replacement_relation": "",
      "reason": ""
    }
  ],
  "queries": [
    {
      "candidate_id": "0",
      "new_query": "",
      "naturalization_reason": ""
    }
  ]
}

Historical dialogue:
<<DIALOGUE>>

Historical constraints:
<<HISTORICAL_CONSTRAINTS>>

Revision anchors:
<<REVISION_ANCHORS>>
""".strip()



VERIFY_UNITE_PROMPT = """Your task is to judge, based on the given historical dialogue, historical constraints, and user's subsequent query, whether the subsequent query satisfies the definition of the target category (EXPECTED_LABEL). If it does, output the target category as label; otherwise, output "Reject".

## Input Objects:
1. Historical dialogue: The complete multi-turn dialogue context before the subsequent query.
2. Historical constraints historical_constraints: Contains completed_main_tasks, current_main_task, and active_constraints. active_constraints are effective historical constraints.
3. User subsequent query list: Each candidate contains candidate_id, subsequent_query, claimed_historical_constraint, and claimed_new_constraint.
   - claimed_historical_constraint: The historical constraint that the generator claims is conflicted with, containing source, level, dimension, and value.
   - claimed_new_constraint: The new constraint that the generator claims is introduced by the subsequent query, containing source, level, dimension, and value.

## Constraint Definitions
1. Explicit constraints: Requirements directly expressed by the user in the subsequent query and clearly identifiable from the literal text. They include but are not limited to topic, object scope, language, format, length, style, audience, output form, technology choice, input data, data source, method, and specific content restrictions.
2. Implicit constraints: Requirements not directly stated by the user in the subsequent query but naturally and stably inferable from the task goal, task type, usage scenario, entity attributes, domain norms, or common sense. Requirements already explicitly stated by the user must not be repeated as implicit constraints.
   
## Classification Definitions
### 1. Normal Dialogue
First understand the current main task, current subtask, historical explicit constraints, and historical implicit constraints based on the historical dialogue. If the subsequent query is compatible with the constraints that are still effective in the history and does not constitute a conflict that needs to be resolved, classify it as Normal Dialogue.
Common forms include:
- Normal continuation: A new request that naturally continues the previous topic or task
- Normal supplementation: A reasonable supplement or refinement request for previous content
- Normal task switch: Switching to a brand-new task request

### 2. Explicit Revision
Within the same task or a naturally continued task, if the subsequent query clearly provides a new explicit constraint and forms a replacement, override, or incompatibility relation with a still-effective explicit constraint in the history, classify it as Explicit Revision.
Corresponding conflict path:
- H_E_N_E: historical explicit constraint conflicts with subsequent new explicit constraint

### 3. Implicit Conflict
Within the same task or a naturally continued task, the subsequent query does not explicitly state that previous requirements should be revised, but it introduces a new constraint that is incompatible with a still-effective constraint in the history, thereby forming a conflict. A conflict exists only when two constraints cannot be satisfied simultaneously; "different", "more detailed", "supplement", and "extension" do not equal a conflict.
Acceptable conflict paths include:
- H_E_N_I: historical explicit constraint conflicts with subsequent new implicit constraint
- H_I_N_E: historical implicit constraint conflicts with subsequent new explicit constraint
- H_I_N_I: historical implicit constraint conflicts with subsequent new implicit constraint

## Verification Steps
1. In "task_judge", output exactly one of these values only: "current_main_task", "current_subtask", or "new_task". Use "current_main_task" when the subsequent query continues the current main task but not the current subtask. Use "current_subtask" when it continues the current subtask. Use "new_task" when it starts a brand-new task. If it is "new_task", directly classify it as Normal Dialogue and do not proceed with the later steps. If it is "current_main_task", follow the historical constraints of the main task. If it is "current_subtask", follow the historical constraints of the subtask.
2. In "history", output the effective constraints extracted from historical constraints for the corresponding level. If it continues the main task, consider only constraints whose level is "current_main_task"; if it continues the subtask, consider only constraints whose level is "current_subtask".
3. If claimed_new_constraint is not empty, strictly place claimed_new_constraint into the corresponding new_explicit_constraints or new_implicit_constraints according to the value of its source.
4. Further extract additional new_explicit_constraints and new_implicit_constraints from the subsequent query according to the constraint definitions.
  - Be careful not to duplicate claimed_new_constraint.
  - dimension and value must be concise, stable, readable phrases in English rather than complex sentences.
  - Each implicit constraint in new_implicit_constraints must contain evidence_span. evidence_span must be in English and should briefly describe the most critical wording in the subsequent query that supports the implicit inference.
5. In "reasoning", write a very detailed verification process: according to the classification definitions, analyze whether the constraints of the subsequent query are compatible with the historical effective constraints; if there is a conflict, explain whether the conflict occurs between explicit constraints or implicit constraints, and output the corresponding conflict path in "channel".
6. "label": judge whether it matches the target category. If it does, output the target category; otherwise output "Reject".
7. "confidence": output the confidence for the label judgement. If label is "Reject", confidence must be 0; otherwise confidence must be an integer from 1 to 100.

## Notes
All text output must be in English.

Output only valid JSON. Do not output explanatory text.

## Output Format
{
  "results": [
    {
      "candidate_id": "0",
      "task_judge": "current_main_task / current_subtask / new_task",
      "history": [
        {
          "level": "current_main_task / current_subtask",
          "source": "",
          "dimension": "",
          "value": "",
          "evidence_span": ""
        }
      ],
      "new_explicit_constraints": [
        {
          "level": "current_main_task / current_subtask",
          "dimension": "",
          "value": ""
        }
      ],
      "new_implicit_constraints": [
        {
          "level": "",
          "dimension": "",
          "value": "",
          "evidence_span": ""
        }
      ],
      "reasoning": "",
      "channel": "",
      "label": "",
      "confidence": 0
    }
  ]
}

## Input

Historical dialogue: <<DIALOGUE>>

Historical constraints: <<HISTORICAL_CONSTRAINTS>>

User subsequent query list: <<CANDIDATES>>

Target category: <<EXPECTED_LABEL>>
"""
