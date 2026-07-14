"""LLM adapters for locally authenticated Codex and Claude subscriptions."""

import asyncio
import json
import logging
import os
import signal
import tempfile
from typing import Any, Dict, List

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import _build_display_prompt

logger = logging.getLogger(__name__)

_AUTH_AND_ROUTING_ENV_VARS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
}
_SHUTDOWN_EVENT = None
_SHUTDOWN_POLL_SECONDS = 0.1


def _set_shutdown_event(event: Any) -> None:
    """Set the process-local shutdown event used by worker adapters."""
    global _SHUTDOWN_EVENT
    _SHUTDOWN_EVENT = event


def _shutdown_requested() -> bool:
    return _SHUTDOWN_EVENT is not None and _SHUTDOWN_EVENT.is_set()


async def _terminate_process(process: Any) -> None:
    """Terminate a CLI process group and reap its leader."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
            if killer.returncode:
                process.kill()
    except (FileNotFoundError, ProcessLookupError):
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


class SubscriptionLLM(LLMInterface):
    """Run a model through an authenticated provider CLI."""

    def __init__(self, model_cfg: Any, provider: str):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.retries = model_cfg.retries or 0
        self.retry_delay = model_cfg.retry_delay or 0
        self.provider = provider
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.generate_with_context(
            self.system_message, [{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> str:
        prompt = _build_display_prompt(messages)
        retries = kwargs.get("retries", self.retries) or 0
        retry_delay = kwargs.get("retry_delay", self.retry_delay) or 0
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)

        for attempt in range(retries + 1):
            if _shutdown_requested():
                raise asyncio.CancelledError
            try:
                text, prompt_tokens, completion_tokens = await self._invoke(
                    prompt, effort, system_message
                )
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_calls += 1
                return text
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if attempt == retries:
                    raise
                logger.warning(
                    "%s CLI attempt %d/%d failed: %s",
                    self.provider,
                    attempt + 1,
                    retries + 1,
                    error,
                )
                if _shutdown_requested():
                    raise asyncio.CancelledError
                await asyncio.sleep(retry_delay)

        raise AssertionError("unreachable")

    async def _invoke(
        self, prompt: str, effort: str | None, system_message: str
    ) -> tuple[str, int, int]:
        command = self._command(effort, system_message)
        if _shutdown_requested():
            raise asyncio.CancelledError
        env = os.environ.copy()
        for name in tuple(env):
            if name in _AUTH_AND_ROUTING_ENV_VARS or name.startswith("CLAUDE_CODE_USE_"):
                env.pop(name)
        other_subscription_token = (
            "CLAUDE_CODE_OAUTH_TOKEN" if self.provider == "codex" else "CODEX_ACCESS_TOKEN"
        )
        env.pop(other_subscription_token, None)

        with tempfile.TemporaryDirectory(prefix="openevolve-") as cwd:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                    start_new_session=os.name == "posix",
                )
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"{command[0]} CLI not found; install it and log in before using "
                    f"the {self.provider} backend"
                ) from error

            communication = asyncio.create_task(process.communicate(prompt.encode()))
            try:
                while not communication.done():
                    if _shutdown_requested():
                        raise asyncio.CancelledError
                    await asyncio.wait({communication}, timeout=_SHUTDOWN_POLL_SECONDS)
                stdout, stderr = communication.result()
            except asyncio.CancelledError:
                communication.cancel()
                await asyncio.gather(communication, return_exceptions=True)
                await asyncio.shield(_terminate_process(process))
                raise

        output = stdout.decode(errors="replace")
        error_output = stderr.decode(errors="replace").strip()
        if process.returncode:
            detail = error_output or output.strip() or "no error output"
            raise RuntimeError(
                f"{self.provider} CLI exited with status {process.returncode}: {detail}"
            )

        try:
            return self._parse(output)
        except (AttributeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            detail = error_output or str(error)
            raise RuntimeError(f"{self.provider} CLI returned malformed JSON: {detail}") from error

    def _command(self, effort: str | None, system_message: str) -> list[str]:
        if self.provider == "codex":
            command = [
                "codex",
                "exec",
                "--model",
                self.model,
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--disable",
                "shell_tool",
                "--disable",
                "unified_exec",
                "--disable",
                "hooks",
                "--disable",
                "apps",
                "--disable",
                "goals",
                "--disable",
                "multi_agent",
                "--disable",
                "remote_plugin",
                "--disable",
                "plugins",
                "--disable",
                "browser_use",
                "--disable",
                "browser_use_external",
                "--disable",
                "browser_use_full_cdp_access",
                "--disable",
                "computer_use",
                "--disable",
                "in_app_browser",
                "--disable",
                "image_generation",
                "--config",
                'web_search="disabled"',
                "--config",
                "project_doc_max_bytes=0",
                "--config",
                f"developer_instructions={json.dumps(system_message or '')}",
                "--color",
                "never",
                "--json",
            ]
            if effort:
                command.extend(["--config", f'model_reasoning_effort="{effort}"'])
            return [*command, "-"]

        command = [
            "claude",
            "--print",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--safe-mode",
            "--tools",
            "",
            "--system-prompt",
            system_message or "",
            "--no-session-persistence",
        ]
        if effort:
            command.extend(["--effort", effort])
        return command

    def _parse(self, output: str) -> tuple[str, int, int]:
        if self.provider == "claude":
            payload = json.loads(output)
            if not isinstance(payload, dict):
                raise TypeError("response is not an object")
            result = payload["result"]
            if not isinstance(result, str):
                raise TypeError("result is not text")
            usage = payload.get("usage")
            if usage is None:
                usage = {}
            if not isinstance(usage, dict):
                raise TypeError("usage is not an object")
            prompt_tokens = sum(
                int(usage.get(name, 0) or 0)
                for name in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            return result, prompt_tokens, int(usage.get("output_tokens", 0) or 0)

        response = None
        usage: Dict[str, Any] = {}
        for line in output.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise TypeError("event is not an object")
            item = event.get("item")
            if item is None:
                item = {}
            if not isinstance(item, dict):
                raise TypeError("item is not an object")
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                response = item.get("text")
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
                if usage is None:
                    usage = {}
                if not isinstance(usage, dict):
                    raise TypeError("usage is not an object")
        if not isinstance(response, str):
            raise ValueError("missing final agent message")
        return (
            response,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )

    def get_token_usage(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "calls": self.total_calls,
        }


def init_codex(model_cfg: Any) -> SubscriptionLLM:
    """Create a Codex subscription client for ``LLMModelConfig.init_client``."""
    return SubscriptionLLM(model_cfg, "codex")


def init_claude(model_cfg: Any) -> SubscriptionLLM:
    """Create a Claude subscription client for ``LLMModelConfig.init_client``."""
    return SubscriptionLLM(model_cfg, "claude")
