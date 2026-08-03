from __future__ import annotations

from pathlib import Path

import yaml

from src.core.config_loader import load_project_config


def test_load_project_config_resolves_relative_paths_against_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "images").mkdir()
    (config_dir / "out").mkdir()

    config_path = config_dir / "project.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_name": "test-project",
                "dataset_path": "./images",
                "output_path": "./out",
                "label_schema": ["person", "car"],
                "sam3": {},
                "qa": {},
                "llm": {},
                "human_review": {},
            }
        ),
        encoding="utf-8",
    )

    config = load_project_config(config_path)

    assert config.dataset_path == config_dir / "images"
    assert config.output_path == config_dir / "out"
