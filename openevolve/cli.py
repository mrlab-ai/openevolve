"""
Command-line interface for OpenEvolve
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from typing import Dict, List, Optional

from openevolve import OpenEvolve
from openevolve.config import Config, load_config
from openevolve.llm.subscription import init_claude, init_codex
from openevolve.model_profiles import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
)

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    """
    Find the highest-numbered checkpoint_N directory

    Args:
        checkpoints_dir: Directory containing checkpoint_N subdirectories

    Returns:
        Path to the newest checkpoint, or None if there are none
    """
    if not os.path.isdir(checkpoints_dir):
        return None

    candidates = []
    for name in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, name)
        if not os.path.isdir(path):
            continue
        match = re.fullmatch(r"checkpoint_(\d+)", name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        return None

    return max(candidates, key=lambda t: t[0])[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")

    parser.add_argument("initial_program", help="Path to the initial program file")

    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument(
        "--iterations",
        "-i",
        help="Number of iterations to run (additional iterations when resuming)",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--until-iteration",
        "-u",
        help="Run until this absolute iteration number in total, resuming included",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument(
        "--auto-resume",
        "-r",
        action="store_true",
        help="Resume from the newest checkpoint in the output directory, if any exists",
    )

    parser.add_argument(
        "--backend",
        choices=("codex", "claude"),
        help="Use an authenticated local subscription CLI",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument(
        "--primary-model", "--model", dest="primary_model", help="Primary LLM model name"
    )

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    parser.add_argument("--reasoning-effort", help="Model reasoning effort", default=None)

    args = parser.parse_args()
    if args.backend and args.api_base is not None:
        parser.error("--api-base cannot be used with --backend")
    if args.backend and args.secondary_model is not None:
        parser.error("--secondary-model cannot be used with --backend")
    if args.checkpoint and args.auto_resume:
        parser.error("--auto-resume cannot be used with --checkpoint")
    if args.iterations is not None and args.until_iteration is not None:
        parser.error("--until-iteration cannot be used with --iterations")
    if args.until_iteration is not None and args.until_iteration < 1:
        parser.error("--until-iteration must be at least 1")
    return args


def _apply_llm_overrides(config: Config, args: argparse.Namespace) -> None:
    """Apply CLI model settings to a loaded configuration."""
    if args.backend:
        model = args.primary_model or (
            DEFAULT_CODEX_MODEL if args.backend == "codex" else DEFAULT_CLAUDE_MODEL
        )
        config.llm.primary_model = model
        config.llm.secondary_model = None
        config.llm.reasoning_effort = args.reasoning_effort
        if args.backend == "codex" and config.llm.reasoning_effort is None:
            config.llm.reasoning_effort = DEFAULT_CODEX_REASONING_EFFORT
        config.llm.rebuild_models()
        config.llm.update_model_params(
            {
                "init_client": init_codex if args.backend == "codex" else init_claude,
                "system_message": config.prompt.system_message,
            },
            overwrite=True,
        )
        print(f"Using {args.backend} subscription backend with model: {model}")
        return

    if args.api_base:
        config.llm.api_base = args.api_base
        print(f"Using API base: {config.llm.api_base}")

    if args.primary_model:
        config.llm.primary_model = args.primary_model
        print(f"Using primary model: {config.llm.primary_model}")

    if args.secondary_model:
        config.llm.secondary_model = args.secondary_model
        print(f"Using secondary model: {config.llm.secondary_model}")

    if args.primary_model or args.secondary_model:
        config.llm.rebuild_models()
        print("Applied CLI model overrides - active models:")
        for i, model in enumerate(config.llm.models):
            print(f"  Model {i+1}: {model.name} (weight: {model.weight})")

    if args.reasoning_effort is not None:
        config.llm.reasoning_effort = args.reasoning_effort
        config.llm.update_model_params({"reasoning_effort": args.reasoning_effort}, overwrite=True)
        print(f"Using reasoning effort: {args.reasoning_effort}")


async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    # Check if files exist
    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    # Load base config from file or defaults
    config = load_config(args.config, ignore_api_keys=bool(args.backend))

    _apply_llm_overrides(config, args)

    # Initialize OpenEvolve
    try:
        openevolve = OpenEvolve(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config=config,
            output_dir=args.output,
        )

        # Resolve which checkpoint to resume from
        checkpoint_path = args.checkpoint
        if args.auto_resume:
            checkpoint_path = find_latest_checkpoint(
                os.path.join(openevolve.output_dir, "checkpoints")
            )
            if checkpoint_path is None:
                print("No checkpoints found - starting a fresh run")

        # Load from checkpoint if specified
        completed_iterations = 0
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                print(f"Error: Checkpoint directory '{checkpoint_path}' not found")
                return 1
            print(f"Loading checkpoint from {checkpoint_path}")
            openevolve.database.load(checkpoint_path)
            completed_iterations = openevolve.database.last_iteration
            print(f"Checkpoint loaded successfully (iteration {completed_iterations})")

        # Translate an absolute iteration target into a relative iteration count
        iterations = args.iterations
        if args.until_iteration is not None:
            iterations = args.until_iteration - completed_iterations
            if iterations <= 0:
                print(
                    f"Already at iteration {completed_iterations}, "
                    f"which meets the target of {args.until_iteration} - nothing to do"
                )
                return 0
            print(f"Running {iterations} iterations to reach a total of {args.until_iteration}")

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = await openevolve.run(
            iterations=iterations,
            target_score=args.target_score,
            checkpoint_path=checkpoint_path,
        )

        # Get the checkpoint path
        latest_checkpoint = find_latest_checkpoint(
            os.path.join(openevolve.output_dir, "checkpoints")
        )

        print(f"\nEvolution complete!")
        print(f"Best program metrics:")
        for name, value in best_program.metrics.items():
            # Handle mixed types: format numbers as floats, others as strings
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint} (or --auto-resume)")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
