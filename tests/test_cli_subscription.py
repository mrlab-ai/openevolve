import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import openevolve.process_parallel as process_parallel
from openevolve.cli import _apply_llm_overrides, parse_args
from openevolve.config import Config, LLMModelConfig, load_config
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.llm.subscription import (
    SubscriptionLLM,
    _shutdown_requested,
    _set_shutdown_event,
    init_claude,
    init_codex,
)
from openevolve.model_profiles import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
)


def _args(**overrides):
    values = {
        "backend": None,
        "api_base": None,
        "primary_model": None,
        "secondary_model": None,
        "reasoning_effort": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class CLISubscriptionTests(unittest.TestCase):
    def test_subscription_load_ignores_yaml_api_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                """llm:
  api_key: ${MISSING_SHARED_KEY}
  models:
    - name: api-model
      api_key: ${MISSING_MODEL_KEY}
"""
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "MISSING_"):
                    load_config(config_path)
                config = load_config(config_path, ignore_api_keys=True)

        self.assertIsNone(config.llm.api_key)
        self.assertEqual([model.name for model in config.llm.models], ["api-model"])
        self.assertIsNone(config.llm.models[0].api_key)

    def test_subscription_defaults_apply_to_every_role(self):
        cases = (
            ("codex", DEFAULT_CODEX_MODEL, DEFAULT_CODEX_REASONING_EFFORT, init_codex),
            ("claude", DEFAULT_CLAUDE_MODEL, None, init_claude),
        )
        for backend, expected_model, expected_effort, initializer in cases:
            with self.subTest(backend=backend), patch("builtins.print"):
                config = Config()
                _apply_llm_overrides(config, _args(backend=backend))

            for models in (
                config.llm.models,
                config.llm.evaluator_models,
                config.llm.repair_models,
            ):
                self.assertEqual(len(models), 1)
                self.assertEqual(models[0].name, expected_model)
                self.assertEqual(models[0].reasoning_effort, expected_effort)
                self.assertEqual(models[0].system_message, config.prompt.system_message)
                self.assertIs(models[0].init_client, initializer)

    def test_subscription_cli_values_replace_yaml_models(self):
        config = Config()
        config.llm.primary_model = "yaml-primary"
        config.llm.secondary_model = "yaml-secondary"
        config.llm.reasoning_effort = "low"

        with patch("builtins.print"):
            _apply_llm_overrides(
                config,
                _args(backend="codex", primary_model="custom", reasoning_effort="xhigh"),
            )

        self.assertEqual(config.llm.primary_model, "custom")
        self.assertIsNone(config.llm.secondary_model)
        for models in (
            config.llm.models,
            config.llm.evaluator_models,
            config.llm.repair_models,
        ):
            self.assertEqual([model.name for model in models], ["custom"])
            self.assertEqual(models[0].reasoning_effort, "xhigh")
            self.assertIs(models[0].init_client, init_codex)

    def test_subscription_models_survive_worker_serialization(self):
        config = Config()
        with patch("builtins.print"):
            _apply_llm_overrides(config, _args(backend="codex"))

        controller = process_parallel.ProcessParallelController(config, "unused", None)
        serialized = controller._serialize_config(config)
        process_parallel._worker_init(
            serialized, "unused", shutdown_event=controller.shutdown_event
        )
        try:
            for models in (
                process_parallel._worker_config.llm.models,
                process_parallel._worker_config.llm.evaluator_models,
                process_parallel._worker_config.llm.repair_models,
            ):
                ensemble = LLMEnsemble(models)
                self.assertEqual(len(ensemble.models), 1)
                self.assertIsInstance(ensemble.models[0], SubscriptionLLM)
        finally:
            _set_shutdown_event(None)

    def test_controller_shutdown_reaches_parent_subscription_client(self):
        controller = process_parallel.ProcessParallelController(Config(), "unused", None)
        try:
            self.assertFalse(_shutdown_requested())
            controller.request_shutdown()
            self.assertTrue(_shutdown_requested())
        finally:
            controller.stop()
        self.assertFalse(_shutdown_requested())

    def test_subscription_config_starts_spawn_worker(self):
        config = Config()
        config.max_tasks_per_child = 1
        config.evaluator.parallel_evaluations = 1
        with patch("builtins.print"):
            _apply_llm_overrides(config, _args(backend="codex"))

        controller = process_parallel.ProcessParallelController(config, "unused", None)
        controller.start()
        try:
            self.assertEqual(controller.executor.submit(len, [1]).result(timeout=10), 1)
        finally:
            controller.stop()

    def test_api_overrides_keep_existing_models_and_apply_reasoning(self):
        config = Config()
        config.llm.models = [LLMModelConfig(name="evolve", reasoning_effort="low")]
        config.llm.evaluator_models = [LLMModelConfig(name="evaluate", reasoning_effort="medium")]
        config.llm.repair_models = [LLMModelConfig(name="repair")]

        with patch("builtins.print"):
            _apply_llm_overrides(config, _args(reasoning_effort="high"))

        self.assertEqual([model.name for model in config.llm.models], ["evolve"])
        self.assertEqual([model.name for model in config.llm.evaluator_models], ["evaluate"])
        self.assertEqual([model.name for model in config.llm.repair_models], ["repair"])
        for models in (
            config.llm.models,
            config.llm.evaluator_models,
            config.llm.repair_models,
        ):
            self.assertEqual(models[0].reasoning_effort, "high")
            self.assertIsNone(models[0].init_client)

    def test_model_alias_parses_and_backend_conflicts_fail(self):
        with patch.object(
            sys,
            "argv",
            [
                "openevolve-run",
                "program.py",
                "evaluate.py",
                "--backend",
                "codex",
                "--model",
                "custom",
                "--reasoning-effort",
                "xhigh",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.primary_model, "custom")
        self.assertEqual(args.reasoning_effort, "xhigh")

        for incompatible in (
            ["--api-base", "https://example.test/v1"],
            ["--secondary-model", "secondary"],
        ):
            with (
                self.subTest(incompatible=incompatible),
                patch.object(
                    sys,
                    "argv",
                    [
                        "openevolve-run",
                        "program.py",
                        "evaluate.py",
                        "--backend",
                        "codex",
                        *incompatible,
                    ],
                ),
                self.assertRaises(SystemExit) as error,
            ):
                parse_args()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
