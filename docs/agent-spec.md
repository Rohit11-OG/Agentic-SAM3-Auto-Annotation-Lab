# Project: Agentic SAM3 Auto‑Annotation Lab

## 1. Overview

Build an **agentic auto-annotation system** that uses **SAM 3** plus LLM-based agents to automatically generate, check, and refine image annotations with minimal human work.

Core ideas:

- Use **SAM 3**’s concept segmentation (text / exemplar prompts) to segment objects for multiple classes in arbitrary images.
- Wrap SAM 3 inside **conversational agents** that:
  - Annotate images.
  - Check annotation quality.
  - Ask each other for retries or clarifications.
  - Escalate hard cases to a human.
- Treat the whole workflow like a **chatroom of agents** rather than a fixed one-pass script.

Target usage:

- Input: folder of images + label schema (class list / ontology).
- Output: dataset in YOLO format with masks + labels + QA metadata.

The system should be **extensible, framework-agnostic**, and easy to plug into tools like CVAT / Label Studio / Roboflow later.

---

## 2. Goals

### Functional goals

- Automatically **annotate images at scale** using SAM 3 with text or exemplar prompts.
- Run **multi-agent QA** on generated annotations:
  - Geometry checks (area, overlap, etc.).
  - Semantic checks (caption vs labels, expected counts).
  - Consistency checks over similar images.
- Allow agents to **freely interact** (chat-style) and decide when to retry SAM 3, when to accept, and when to send to human review.
- Export final annotations to standard formats and generate basic QA reports for each batch.

### Non-functional goals

- Pure Python backend; no requirement on any specific agent framework (LangChain, LangGraph, CrewAI, etc.), but the design should map easily to one.
- Modular architecture (agents, tools, data models, orchestrator separated).
- Config-driven: datasets, labels, thresholds, and SAM 3 parameters are configurable via YAML/JSON.

---

## 3. High-Level Architecture

**Components:**

1. **Agent Orchestrator / Chatroom**
   - Owns the global message history and controls which agent speaks next.
   - Enforces global rules (max retries, stop conditions).

2. **Annotator Agent (SAM3Agent)**
   - Uses SAM 3 (via Ultralytics / official client / Roboflow) to produce masks + labels from text/exemplar prompts.

3. **QA Agent (CurationAgent)**
   - Validates annotations using:
     - Numeric/heuristic checks (area, IoU, overlaps).
     - Optional LLM reasoning (image captions + annotation metadata).

4. **Coordinator / Supervisor Agent (CoordinatorAgent)**
   - Ensures the conversation flows:
     - Decides which agent should respond.
     - Tracks per-image state (NEW → ANNOTATED → QA_RETRY → ACCEPTED / HUMAN_REVIEW).
   - Applies policies like max retries, time budget, risk tiers.

5. **Tool Layer**
   - SAM 3 client (image → masks via text/exemplar prompt).
   - Image captioning / VLM (optional, for semi-automatic class discovery / checking).
   - Geometry utilities (area, IoU, overlaps, class-wise stats).
   - Exporter (YOLO).
   - Storage & logging (annotations, QA scores, conversation logs).

---

## 4. Data Model

Use Pydantic dataclasses or plain typed dataclasses.

### 4.1 Core entities

```python
ImageId = str
MaskId = str
ClassId = str
AgentName = str
```

#### `ProjectConfig`

- `project_name: str`
- `dataset_path: Path`
- `output_path: Path`
- `label_schema: List[ClassId]`  (e.g. `['person', 'car', 'dog']`)
- `sam3_model_name: str`  (e.g. `"sam3_b"` or Ultralytics equivalent)
- `max_retries: int`
- `min_mask_area: float` (relative to image area, e.g. `0.001`)
- `max_mask_area: float`
- `qa_iou_threshold: float`
- `qa_confidence_threshold: float`
- `enable_captioning: bool`
- `llm_model_name: str` (for QA reasoning)
- `human_review_policy: {"max_retries": int, "enabled": bool}`

#### `ImageRecord`

- `id: ImageId`
- `path: Path`
- `width: int`
- `height: int`
- `meta: Dict[str, Any]` (optional: tags, source, etc.)

#### `MaskRecord`

- `mask_id: MaskId`
- `image_id: ImageId`
- `class_id: ClassId`
- `polygon: List[Tuple[int, int]]` or RLE
- `bbox: Tuple[int, int, int, int]`
- `area: float` (pixels or normalized)
- `confidence: float`
- `source: Literal['sam3', 'human', 'other']`
- `version: int` (increment on retries)

#### `AnnotationBundle`

- `image: ImageRecord`
- `masks: List[MaskRecord]`
- `qa_result: Optional[QAResult]`
- `status: Literal['NEW', 'ANNOTATED', 'QA_RETRY', 'ACCEPTED', 'HUMAN_REVIEW']`
- `history: List[ConversationMessage]`

#### `QAResult`

- `image_id: ImageId`
- `quality_score: float` (0–1)
- `issues: List[str]` (human-readable summary)
- `decision: Literal['ACCEPT', 'RETRY_WITH_HINTS', 'HUMAN_REVIEW']`
- `hints: Dict[str, Any]` (per-class or per-mask suggestions, e.g. `{"car": {"min_area": 0.01}}`)

---

## 5. Agent Chatroom & Message Schema

### 5.1 Conversation message

```python
class ConversationMessage(BaseModel):
    message_id: str
    image_id: Optional[ImageId]
    sender: AgentName  # 'SAM3Agent', 'CurationAgent', 'CoordinatorAgent'
    role: Literal['system', 'agent', 'tool']
    content: str       # natural language
    actions: List[Dict[str, Any]]  # structured actions parsed from content
    timestamp: datetime
```

Example `actions`:

- Annotator:
  - `{"type": "ANNOTATE", "image_id": "...", "classes": ["car", "person"], "mode": "text_prompt"}`
  - `{"type": "ANNOTATION_RESULT", "image_id": "...", "masks": [...]}`
- QA:
  - `{"type": "QA_DECISION", "image_id": "...", "decision": "RETRY_WITH_HINTS", "qa_result": {...}}`
- Coordinator:
  - `{"type": "SET_STATUS", "image_id": "...", "status": "ACCEPTED"}`  
  - `{"type": "REQUEST_RETRY", "image_id": "...", "retry_count": 1}`

---

## 6. Agents

### 6.1 Annotator Agent (`SAM3Agent`)

**Role:** Generate segmentation masks for target classes using SAM 3.

**Inputs:**

- `ImageRecord`
- target `label_schema`
- optional hints from QA (e.g. increase max area, look for second instance, relax thresholds).

**Capabilities:**

- Use **SAM 3 with text prompts**:
  - For each class, call SAM 3 concept segmentation to find all instances.
- Optionally use **exemplar prompts**:
  - Use existing masks/bboxes as prompts for refining segmentation.
- Output `MaskRecord` list with confidence and area.

**System prompt skeleton:**

> You are SAM3Agent.  
> Your job is to annotate images using SAM 3.  
> You receive instructions and hints from CurationAgent and CoordinatorAgent.  
> For each image and class, you should:
> 1. Decide which prompt type to use (text or exemplar).
> 2. Call the SAM 3 tool with those prompts.
> 3. Return structured actions including `"ANNOTATION_RESULT"` with masks and metadata.
> You must follow label_schema and ignore classes not in schema.

**Tools used:**

- `tool_sam3_segment(image_path, class_name, mode, params) -> List[RawMask]`
- `tool_convert_masks(raw_masks, image) -> List[MaskRecord]`

---

### 6.2 QA / Curation Agent (`CurationAgent`)

**Role:** Evaluate the quality of SAM 3 annotations and decide whether to accept, retry with hints, or escalate to human review.

**Inputs:**

- `AnnotationBundle` (current masks + image metadata).
- Optional caption / textual description of the image.

**Capabilities:**

- Run **heuristic checks**:
  - Area filter: `min_mask_area <= area <= max_mask_area` for each class.
  - Overlap anomalies: IoU between masks; detect suspicious overlaps (e.g., car fully inside car, duplicate segments).
  - Count vs expectations: If captions say “two dogs” and we only have one dog mask, flag.
- Run **LLM reasoning**:
  - Given caption + summary of masks, ask: “Does this annotation look plausible?”  
- Produce a `QAResult` with:
  - `quality_score`
  - `issues`
  - `decision` (`ACCEPT`, `RETRY_WITH_HINTS`, `HUMAN_REVIEW`)
  - structured `hints` for SAM3Agent (e.g., by class, by region).

**System prompt skeleton:**

> You are CurationAgent.  
> Your job is to check whether annotations are good enough.  
> Check: mask sizes, overlaps, class counts, and alignment with captions (if available).  
> If issues are minor and fixable via SAM 3, return decision `RETRY_WITH_HINTS` and explicit hints.  
> If issues are large or ambiguous after several retries, return `HUMAN_REVIEW`.  
> Otherwise, return `ACCEPT`.  
> Always include a numeric quality_score (0 to 1) and short human-readable issues list.

---

### 6.3 Coordinator Agent (`CoordinatorAgent`)

**Role:** Manage the agent conversation and per-image state.

**Responsibilities:**

- Decide which agent should respond next based on current messages and image status:
  - If image is `NEW` → ask `SAM3Agent` to annotate.
  - If `ANNOTATED` & no QA yet → ask `CurationAgent` to check.
  - If `QA_RETRY` & retries < `max_retries` → ask `SAM3Agent` to re-annotate with hints.
  - If `QA_RETRY` & retries >= `max_retries` → mark `HUMAN_REVIEW`.
- Enforce global constraints:
  - `max_retries` per image.
  - Global timeouts or batch size limits.
- Update `AnnotationBundle.status` based on QA decisions.

**System prompt skeleton:**

> You are CoordinatorAgent.  
> You control the conversation between SAM3Agent and CurationAgent.  
> For each image, you:
> - Track its status: NEW → ANNOTATED → QA_RETRY → ACCEPTED / HUMAN_REVIEW.  
> - Decide which agent to call next.  
> - Stop after a decision is made or retries reach the limit.  
> You produce `SET_STATUS` and `REQUEST_RETRY` actions, and you may send high-level comments to the other agents.

---

## 7. Orchestrator Flow

Implement as a central loop (`orchestrator.py`).

### 7.1 Batch-level flow

For each batch:

1. Load `ProjectConfig` and image list.
2. For each image, create an initial `AnnotationBundle` with status `NEW`.
3. For each bundle:
   - While status not in `['ACCEPTED', 'HUMAN_REVIEW']`:
     1. Coordinator reads conversation history + bundle, emits a message with actions (`REQUEST_ANNOTATION`, `REQUEST_QA`, or `REQUEST_RETRY`).
     2. Appropriate agent reads history + bundle + actions, responds with new message and actions.
     3. Tools are invoked as required (SAM3, geometry, etc.).
     4. Bundle is updated (masks, QAResult, status).
4. Export accepted annotations + logs.

### 7.2 Example single-image sequence

1. **Coordinator → SAM3Agent**

   - Message:
     - `content`: "Image 17 is NEW. SAM3Agent, please annotate for classes: car, person."
     - `actions`: `{"type": "REQUEST_ANNOTATION", ...}`

2. **SAM3Agent → Tools → SAM3Agent**

   - Calls `tool_sam3_segment` for `"car"` and `"person"`.  
   - Builds `MaskRecord`s and sends:
     - `content`: "Annotated image 17 with 2 car masks and 1 person mask."
     - `actions`: `{"type": "ANNOTATION_RESULT", ...}`  

3. **Coordinator → CurationAgent**

   - Sees `ANNOTATION_RESULT`, sets bundle status to `ANNOTATED`.  
   - Message: ask QA to evaluate.

4. **CurationAgent**

   - Runs heuristics + LLM reasoning.  
   - Suppose it finds person mask too small & caption says "two people":
     - `decision`: `RETRY_WITH_HINTS`
     - `hints`: `{"person": {"min_instances": 2, "min_area": 0.01}}`
   - Sends message with `QA_DECISION`.

5. **Coordinator**

   - Status: `QA_RETRY`, increments retry count (1).  
   - If retry_count < `max_retries`, asks `SAM3Agent` to retry with hints.

6. **SAM3Agent (retry)**

   - Adjusts SAM3 params (lower threshold, different prompt) and re-runs only for `"person"`.  
   - Sends updated `ANNOTATION_RESULT`.

7. **CurationAgent (second QA)**

   - If now OK, returns `ACCEPT`.  
   - Coordinator sets status `ACCEPTED`.

8. **Export**

   - Final masks + QA metadata exported to chosen format.

---

## 8. Tools Layer – Details

### 8.1 SAM 3 client

Implement wrapper functions around SAM 3 according to chosen library (Ultralytics, official SDK, or Roboflow).

```python
def sam3_segment_text_prompt(
    image_path: Path,
    class_name: str,
    model_name: str,
    params: Dict[str, Any]
) -> List[RawMask]:
    ...


def sam3_segment_exemplar_prompt(
    image_path: Path,
    exemplar_bbox: Tuple[int, int, int, int],
    model_name: str,
    params: Dict[str, Any]
) -> List[RawMask]:
    ...
```

### 8.2 Geometry utilities

```python
def compute_area(mask: np.ndarray) -> float: ...

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float: ...

def summarize_masks(masks: List[MaskRecord]) -> Dict[str, Any]: ...
```

Used by CurationAgent for heuristic checks (tiny masks, overlaps, etc.).

### 8.3 Exporters

- `export_yolo(bundles: List[AnnotationBundle], output_path: Path)`

The exporter reads accepted masks and converts them to YOLO format.

### 8.4 Captioning (optional)

```python
def caption_image(image_path: Path) -> str:
    # call local VLM or external API
    ...
```

Used by CurationAgent to compare expected vs actual object counts and labels.

---

## 9. Repository & File Structure

Suggested structure:

```text
agentic-sam3-annotator/
  AGENTIC_SAM3_ANNOTATION_SPEC.md   # this spec
  pyproject.toml / requirements.txt
  config/
    project_example.yaml
  data/
    images/
    annotations_raw/
    annotations_final/
  src/
    core/
      orchestrator.py
      models.py          # ImageRecord, MaskRecord, AnnotationBundle, ConversationMessage, QAResult
      config_loader.py
      logging_utils.py
    tools/
      sam3_client.py
      geometry.py
      yolo/
        exporter.py
      captioning.py
    agents/
      base_agent.py
      sam3_agent.py          # AnnotatorAgent
      curation_agent.py      # QA agent
      coordinator_agent.py   # Supervisor agent
    ui/
      review_app.py          # optional simple web UI for human review
  notebooks/
    exploration.ipynb        # quick tests for SAM3 / QA logic
  tests/
    test_sam3_client.py
    test_geometry.py
    test_agents.py
```

---

## 10. Configuration Example (`config/project_example.yaml`)

```yaml
project_name: "sam3_auto_annotation_lab"

dataset_path: "./data/images"
output_path: "./data/annotations_final"

label_schema:
  - person
  - car
  - dog

sam3:
  model_name: "sam3_b"
  device: "cuda:0"
  text_prompt_mode: true

qa:
  max_retries: 2
  min_mask_area: 0.001
  max_mask_area: 0.5
  iou_threshold: 0.7
  confidence_threshold: 0.3
  enable_captioning: true

llm:
  model_name: "your-llm-name"

human_review:
  enabled: true
  max_retries: 2
```

---

## 11. Implementation Phases

### Phase 1 – Basic pipeline (no chatroom)

- Implement SAM3 client and geometry utilities.
- Single script: load images → run SAM3 → export YOLO.
- No QA, no agents (just to verify SAM3).

### Phase 2 – Single-agent iteration

- Create `SAM3Agent` + `CurationAgent` as functions, not full chat.
- Sequence: SAM3 → QA → optional retry → export.

### Phase 3 – Multi-agent chatroom

- Implement `ConversationMessage`, message history, `CoordinatorAgent`.
- Implement orchestrator loop that lets agents “talk” to each other through the shared history, as described above.

### Phase 4 – Human-in-the-loop & UI

- Add a simple web UI (FastAPI + React, or Streamlit/Gradio) to:
  - Filter `HUMAN_REVIEW` images.
  - Manually fix masks (or accept/reject).
- Integrate manual edits back into final export.

---

## 12. Agent Interaction Rules

To ensure agents can **"freely interact like humans"** but still stay safe:

- All agents read **the same conversation history** for a given image (last N messages window).
- Each agent has a `should_respond(history, bundle)` function to decide if it should speak.
- Only Coordinator can:
  - Change `status`.
  - Decide when the conversation for an image is done.
- Annotator and QA can address each other in **natural-language comments** but must also emit structured `actions` that the orchestrator parses.

Example rule set:

- `SAM3Agent` speaks when:
  - Coordinator has requested annotation or retry for that image.
- `CurationAgent` speaks when:
  - New `ANNOTATION_RESULT` exists for that image and no QA decision yet.
- `CoordinatorAgent` speaks when:
  - New SAM3 or QA messages exist, and status update is needed.

---

## 13. Possible Extensions

- Use an **object detector** (YOLO) to produce bounding boxes as prompts to SAM 3 for even more precise segmentation (two-stage pipeline).
- Add a **MetricsAgent** that monitors annotation statistics (per-class coverage, noise) and suggests global configuration changes.
- Add an **OntologyAgent** that helps manage and grow label schemas across projects (merging synonyms, etc.) using agentic curation patterns.
- Integrate with cloud storage and job queues (Celery/RQ/Kafka) for large-scale datasets.

---

This spec is meant to be **concrete enough** that an AI coding agent (or you) can:

- Generate the full repo skeleton.
- Implement each agent and tool step by step.
- Extend it as your SAM 3 / labeling needs grow.
