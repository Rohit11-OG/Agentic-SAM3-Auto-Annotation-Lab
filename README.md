# Agentic SAM3 Auto-Annotation Lab

An extensible Python backend for auto-annotating images with a SAM3-style segmentation client wrapped in a multi-agent workflow:

- `SAM3Agent` generates masks from class prompts.
- `CurationAgent` scores and validates quality.
- `CoordinatorAgent` manages retries, acceptance, and escalation.

## Quick Start

1. Create a virtual environment and install dependencies:
   - `pip install -e .[dev]`
2. Copy or edit config:
   - `config/project_example.yaml`
3. Place images in:
   - `data/images`
4. Run:
   - `python -m src.main --config config/project_example.yaml`

## Current State

The project includes:

- Typed data models (`src/core/models.py`)
- Config loading (`src/core/config_loader.py`)
- Orchestrator chatroom loop (`src/core/orchestrator.py`)
- Agent implementations (`src/agents/`)
- Tooling layer (`src/tools/`)
- Exporter for YOLO (`src/tools/yolo/exporter.py`)
- Unit tests (`tests/`)

The SAM3 client supports two backends:

- `mock`: deterministic local masks for free/offline development.
- `sam3_api`: calls a hosted SAM3-style API endpoint from `src/tools/sam3_client.py`.

Hosted APIs are usually paid after free trial credits. Keep `backend: mock` to use the project fully without paid inference.
