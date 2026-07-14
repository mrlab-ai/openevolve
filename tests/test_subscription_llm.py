import asyncio
import json
import os
import pickle
import signal
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openevolve.llm.subscription import (
    _set_shutdown_event,
    _terminate_process,
    init_claude,
    init_codex,
)


def config(**overrides):
    values = {
        "name": "test-model",
        "system_message": "Be useful.",
        "reasoning_effort": None,
        "retries": 0,
        "retry_delay": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class Process:
    def __init__(self, stdout=b"", stderr=b"", returncode=0, pid=1234):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = pid
        self.input = None
        self.killed = False
        self.waited = False

    async def communicate(self, value):
        self.input = value
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True


class SubscriptionLLMTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        _set_shutdown_event(None)

    async def test_codex_command_prompt_jsonl_usage_and_clean_environment(self):
        output = "\n".join(
            json.dumps(event)
            for event in (
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "first"},
                },
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "final"},
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
            )
        ).encode()
        process = Process(stdout=output)
        create = AsyncMock(return_value=process)
        client = init_codex(config(reasoning_effort="max"))

        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "api-key",
                    "OPENAI_BASE_URL": "https://openai.invalid",
                    "ANTHROPIC_API_KEY": "api-key",
                    "ANTHROPIC_AUTH_TOKEN": "token",
                    "ANTHROPIC_BASE_URL": "https://anthropic.invalid",
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
                    "CODEX_ACCESS_TOKEN": "codex-subscription-token",
                    "CODEX_API_KEY": "api-key",
                    "KEEP_ME": "yes",
                },
            ),
            patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create),
        ):
            result = await client.generate_with_context(
                "System", [{"role": "assistant", "content": "Earlier"}]
            )

        self.assertEqual(result, "final")
        self.assertEqual(process.input.decode(), "### ASSISTANT\nEarlier\n")
        args, kwargs = create.await_args
        self.assertEqual(args[:2], ("codex", "exec"))
        self.assertIn("read-only", args)
        self.assertIn("--ephemeral", args)
        self.assertIn("--ignore-user-config", args)
        self.assertIn("--ignore-rules", args)
        self.assertEqual(
            {
                args[index + 1]
                for index, value in enumerate(args[:-1])
                if value == "--disable"
            },
            {
                "shell_tool",
                "unified_exec",
                "hooks",
                "apps",
                "goals",
                "multi_agent",
                "remote_plugin",
                "plugins",
                "browser_use",
                "browser_use_external",
                "browser_use_full_cdp_access",
                "computer_use",
                "in_app_browser",
                "image_generation",
            },
        )
        self.assertIn("project_doc_max_bytes=0", args)
        self.assertIn('developer_instructions="System"', args)
        self.assertTrue(kwargs["start_new_session"])
        self.assertIn('model_reasoning_effort="max"', args)
        self.assertIn('web_search="disabled"', args)
        self.assertEqual(args[-1], "-")
        self.assertNotIn("timeout", kwargs)
        self.assertTrue(kwargs["cwd"].startswith("/tmp/openevolve-"))
        self.assertFalse(os.path.exists(kwargs["cwd"]))
        self.assertEqual(kwargs["env"]["KEEP_ME"], "yes")
        self.assertTrue(
            {
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CODEX_API_KEY",
            }.isdisjoint(kwargs["env"])
        )
        self.assertEqual(kwargs["env"]["CODEX_ACCESS_TOKEN"], "codex-subscription-token")
        self.assertEqual(
            client.get_token_usage(),
            {
                "model": "test-model",
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
                "calls": 1,
            },
        )

    async def test_claude_safe_command_and_json_usage(self):
        process = Process(
            stdout=json.dumps(
                {
                    "result": "answer",
                    "usage": {
                        "input_tokens": 4,
                        "cache_creation_input_tokens": 5,
                        "cache_read_input_tokens": 6,
                        "output_tokens": 7,
                    },
                }
            ).encode()
        )
        create = AsyncMock(return_value=process)
        client = init_claude(config(reasoning_effort="high"))

        with (
            patch.dict(
                os.environ,
                {
                    "CLAUDE_CODE_OAUTH_TOKEN": "claude-subscription-token",
                    "CODEX_ACCESS_TOKEN": "codex-subscription-token",
                },
            ),
            patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create),
        ):
            self.assertEqual(await client.generate("hello"), "answer")

        args, kwargs = create.await_args
        self.assertEqual(process.input.decode(), "### USER\nhello\n")
        self.assertEqual(args[0], "claude")
        self.assertIn("--print", args)
        self.assertIn("--safe-mode", args)
        self.assertIn("--no-session-persistence", args)
        self.assertEqual(args[args.index("--tools") + 1], "")
        self.assertEqual(args[args.index("--system-prompt") + 1], "Be useful.")
        self.assertEqual(args[args.index("--effort") + 1], "high")
        self.assertEqual(kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"], "claude-subscription-token")
        self.assertNotIn("CODEX_ACCESS_TOKEN", kwargs["env"])
        self.assertEqual(client.get_token_usage()["prompt_tokens"], 15)
        self.assertEqual(client.get_token_usage()["completion_tokens"], 7)

    async def test_failure_retries_without_timeout_and_surfaces_stderr(self):
        failed = Process(stderr=b"please log in", returncode=1)
        succeeded = Process(stdout=json.dumps({"result": "ok", "usage": {}}).encode())
        create = AsyncMock(side_effect=[failed, succeeded])
        sleep = AsyncMock()
        client = init_claude(config(retries=1, retry_delay=2))

        with (
            patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create),
            patch("openevolve.llm.subscription.asyncio.sleep", sleep),
        ):
            self.assertEqual(await client.generate("hello"), "ok")

        self.assertEqual(create.await_count, 2)
        sleep.assert_awaited_once_with(2)
        for call in create.await_args_list:
            self.assertNotIn("timeout", call.kwargs)

        with patch(
            "openevolve.llm.subscription.asyncio.create_subprocess_exec",
            AsyncMock(return_value=failed),
        ):
            with self.assertRaisesRegex(RuntimeError, "please log in"):
                await init_claude(config()).generate("hello")

    async def test_cancellation_kills_and_reaps_process(self):
        started = asyncio.Event()
        process = Process()

        async def communicate(value):
            process.input = value
            started.set()
            await asyncio.Future()

        process.communicate = communicate
        create = AsyncMock(return_value=process)
        client = init_codex(config())

        with (
            patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create),
            patch("openevolve.llm.subscription.os.killpg") as kill_group,
        ):
            task = asyncio.create_task(client.generate("hello"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        kill_group.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertTrue(process.waited)

    async def test_shutdown_event_cancels_active_process_group(self):
        started = asyncio.Event()
        shutdown = threading.Event()
        process = Process()

        async def communicate(value):
            process.input = value
            started.set()
            await asyncio.Future()

        process.communicate = communicate
        create = AsyncMock(return_value=process)
        _set_shutdown_event(shutdown)

        with (
            patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create),
            patch("openevolve.llm.subscription.os.killpg") as kill_group,
        ):
            task = asyncio.create_task(init_codex(config()).generate("hello"))
            await started.wait()
            shutdown.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)

        kill_group.assert_called_once_with(process.pid, signal.SIGKILL)
        self.assertTrue(process.waited)

    async def test_windows_termination_uses_taskkill_tree(self):
        process = Process()
        killer = Process()
        create = AsyncMock(return_value=killer)

        with (
            patch("openevolve.llm.subscription.os.name", "nt"),
            patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create),
        ):
            await _terminate_process(process)

        self.assertEqual(
            create.await_args.args,
            ("taskkill", "/F", "/T", "/PID", str(process.pid)),
        )
        self.assertTrue(process.waited)
        self.assertTrue(killer.waited)

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "requires /proc")
    async def test_process_group_termination_kills_child_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            script = (
                "import subprocess, sys, time\n"
                "from pathlib import Path\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "Path(sys.argv[1]).write_text(str(child.pid))\n"
                "time.sleep(60)\n"
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                str(pid_file),
                start_new_session=True,
            )
            try:
                for _ in range(100):
                    if pid_file.exists():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(pid_file.exists())
                child_pid = int(pid_file.read_text())

                await _terminate_process(process)

                for _ in range(100):
                    try:
                        state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
                    except FileNotFoundError:
                        state = None
                    if state in (None, "Z"):
                        break
                    await asyncio.sleep(0.01)
                self.assertIn(state, (None, "Z"))
            finally:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()

    async def test_missing_binary_and_malformed_output_are_actionable(self):
        client = init_codex(config())
        with patch(
            "openevolve.llm.subscription.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError),
        ):
            with self.assertRaisesRegex(RuntimeError, "CLI not found.*install.*log in"):
                await client.generate("hello")

        for factory, output in (
            (init_codex, b"not json"),
            (init_codex, b"[]"),
            (init_claude, b'{"result":"x","usage":[]}'),
        ):
            with (
                self.subTest(output=output),
                patch(
                    "openevolve.llm.subscription.asyncio.create_subprocess_exec",
                    AsyncMock(return_value=Process(stdout=output)),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "malformed JSON"):
                    await factory(config()).generate("hello")

    def test_factories_are_pickleable(self):
        self.assertIs(pickle.loads(pickle.dumps(init_codex)), init_codex)
        self.assertIs(pickle.loads(pickle.dumps(init_claude)), init_claude)


if __name__ == "__main__":
    unittest.main()
