from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.config_loader import load_project_config
from src.core.logging_utils import setup_logging
from src.core.orchestrator import discover_images, run_orchestrator
from src.tools.prompt_interpreter import interpret_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agentic SAM3 auto-annotation pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project_example.yaml"),
        help="Path to YAML project config.",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Override dataset_path.")
    parser.add_argument("--output", type=Path, default=None, help="Override output_path.")
    parser.add_argument("--workers", type=int, default=None, help="Override max_workers.")
    parser.add_argument("--max-retries", type=int, default=None, help="Override QA max_retries.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='Free-text annotation request, e.g. "annotate cars and people in street photos".',
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read annotation prompt from stdin interactively.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered images and config, do not run agents.",
    )
    return parser.parse_args()


def _apply_overrides(config, args: argparse.Namespace):
    if args.dataset is not None:
        config.dataset_path = args.dataset
    if args.output is not None:
        config.output_path = args.output
    if args.workers is not None:
        config.max_workers = max(1, args.workers)
    if args.max_retries is not None:
        config.max_retries = max(0, args.max_retries)
    return config


def _apply_prompt(config, args: argparse.Namespace) -> None:
    user_prompt = args.prompt
    if user_prompt is None and args.interactive:
        try:
            print("Enter annotation prompt (e.g. 'annotate cars and people'):")
            user_prompt = input("> ").strip()
        except EOFError:
            user_prompt = None
    if not user_prompt:
        return

    plan = interpret_prompt(user_prompt, fallback_schema=config.label_schema)
    config.user_prompt = plan.raw_input
    if plan.classes:
        config.label_schema = plan.classes
    config.per_class_prompt = plan.per_class_prompt

    import json
    log = logging.getLogger(__name__)
    log.info("Class Label: %s", json.dumps(plan.classes))
    for cls in plan.classes:
        prompt_text = plan.per_class_prompt.get(cls, cls)
        log.info("SAM3 Text Prompt: \"%s\"", prompt_text)
    for note in plan.notes:
        log.info("  note: %s", note)


def main() -> int:
    args = parse_args()
    config = load_project_config(args.config)
    config = _apply_overrides(config, args)
    _apply_prompt(config, args)

    config.output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(config.output_path / "logs", level=getattr(logging, args.log_level))

    log = logging.getLogger(__name__)

    if args.dry_run:
        images = discover_images(config.dataset_path, exclude=config.output_path)
        log.info(
            "DRY-RUN: project=%s dataset=%s images=%d classes=%s workers=%d retries=%d",
            config.project_name,
            config.dataset_path,
            len(images),
            config.label_schema,
            config.max_workers,
            config.max_retries,
        )
        for im in images:
            log.info("  %s %s %dx%d", im.id, im.path.name, im.width, im.height)
        return 0

    bundles = run_orchestrator(config)
    accepted = sum(1 for bundle in bundles if bundle.status == "ACCEPTED")
    human_review = sum(1 for bundle in bundles if bundle.status == "HUMAN_REVIEW")
    log.info(
        "Run complete. total=%d accepted=%d human_review=%d",
        len(bundles),
        accepted,
        human_review,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
