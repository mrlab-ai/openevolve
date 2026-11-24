import datetime
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openevolve.llm.subscription import _set_shutdown_event, init_codex
from openevolve.llm.usage_limits import (
    UsageLimitHandoff,
    UsageLimitHit,
    codex_diagnostics,
    match_limit,
    parse_reset_epoch,
    plan_limit_response,
)


def config():
    return SimpleNamespace(
        name="gpt-6-astra", system_message="s", reasoning_effort="max",
        retries=0, retry_delay=0,
    )


class Process:
    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self.stdout, self.stderr, self.returncode, self.pid = stdout, stderr, returncode, 1

    async def communicate(self, value):
        return self.stdout, self.stderr

    def kill(self):
        pass

    async def wait(self):
        pass


def ok_output(text="ok"):
    return "\n".join(json.dumps(e) for e in (
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    )).encode()


LIMIT_STDERR = (b"ERROR: You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage "
                b"to purchase more credits or try again at Sep 15th, 2026 3:24 AM.\n")


class ParseTests(unittest.TestCase):
    def test_codex_prose_reset(self):
        epoch = parse_reset_epoch(LIMIT_STDERR.decode())
        self.assertEqual(epoch, int(datetime.datetime(2026, 9, 15, 3, 24).timestamp()))

    def test_json_resets_at_prefers_earliest_future(self):
        now = 1_800_000_000
        blob = json.dumps({"resets_at": now + 500}) + json.dumps({"resets_at": now + 90000})
        self.assertEqual(parse_reset_epoch(blob, now=now), now + 500)

    def test_resets_in_duration(self):
        self.assertEqual(parse_reset_epoch("Resets in 1h2m3s.", now=1000), 1000 + 3723)

    def test_diagnostics_ignore_agent_text_and_reconnects(self):
        stdout = "\n".join(json.dumps(e) for e in (
            {"type": "item.completed", "item": {"type": "agent_message",
                                                "text": "rate limit 429 quota"}},
        ))
        stderr = "ERROR: Reconnecting... 1/5 (rate limit)\n"
        self.assertFalse(match_limit(codex_diagnostics(stdout, stderr)))

    def test_turn_failed_event_matches(self):
        stdout = json.dumps({"type": "turn.failed",
                             "error": {"message": "You've hit your usage limit"}})
        self.assertTrue(match_limit(codex_diagnostics(stdout, "")))


class PlanTests(unittest.TestCase):
    def test_near_reset_waits(self):
        with patch.dict(os.environ, {"OPENEVOLVE_HANDOFF_FILE": "/x"}):
            action, delay = plan_limit_response(UsageLimitHit("m", 1600), 1, now=1000)
        self.assertEqual(action, "wait")
        self.assertEqual(delay, 600 + 120)

    def test_far_reset_hands_off(self):
        with patch.dict(os.environ, {"OPENEVOLVE_HANDOFF_FILE": "/x"}):
            action, at = plan_limit_response(UsageLimitHit("m", 1000 + 86400), 1, now=1000)
        self.assertEqual((action, at), ("handoff", 1000 + 86400 + 120))

    def test_without_handoff_far_reset_waits_capped(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENEVOLVE_HANDOFF_FILE", None)
            action, delay = plan_limit_response(UsageLimitHit("m", 1000 + 86400), 1, now=1000)
        self.assertEqual((action, delay), ("wait", 21600))

    def test_unknown_reset_eventually_hands_off(self):
        with patch.dict(os.environ, {"OPENEVOLVE_HANDOFF_FILE": "/x"}):
            self.assertEqual(plan_limit_response(UsageLimitHit("m", None), 1, now=0)[0], "wait")
            self.assertEqual(plan_limit_response(UsageLimitHit("m", None), 13, now=0)[0], "handoff")


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        _set_shutdown_event(None)

    async def test_short_limit_is_waited_out_and_same_call_retried(self):
        soon = int(time.time()) + 60
        limited = Process(stdout=b"", stderr=f"ERROR: usage limit. Resets in 1m0s\n".encode(),
                          returncode=1)
        create = AsyncMock(side_effect=[limited, Process(stdout=ok_output("done"))])
        sleep = AsyncMock()
        with tempfile.TemporaryDirectory() as d, \
                patch.dict(os.environ, {"OPENEVOLVE_HANDOFF_FILE": f"{d}/h.json"}), \
                patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create), \
                patch("openevolve.llm.subscription._sleep_unless_shutdown", sleep):
            result = await init_codex(config()).generate("p")
            self.assertFalse(os.path.exists(f"{d}/h.json"))
        self.assertEqual(result, "done")
        self.assertEqual(create.await_count, 2)
        sleep.assert_awaited_once()

    async def test_far_limit_writes_handoff_and_raises(self):
        create = AsyncMock(return_value=Process(stderr=LIMIT_STDERR, returncode=1))
        with tempfile.TemporaryDirectory() as d, \
                patch.dict(os.environ, {"OPENEVOLVE_HANDOFF_FILE": f"{d}/h.json"}), \
                patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create), \
                patch("openevolve.llm.usage_limits.time.time", return_value=1_700_000_000):
            with self.assertRaises(UsageLimitHandoff):
                await init_codex(config()).generate("p")
            record = json.load(open(f"{d}/h.json"))
        self.assertEqual(record["reason"], "usage_limit")
        self.assertEqual(
            record["resume_at"],
            int(datetime.datetime(2026, 9, 15, 3, 24).timestamp()) + 120,
        )

    async def test_limit_does_not_consume_retries(self):
        # retries=0: an ordinary failure would raise immediately.
        limited = Process(stderr=b"ERROR: rate limit exceeded\n", returncode=1)
        create = AsyncMock(side_effect=[limited, limited, Process(stdout=ok_output())])
        with patch.dict(os.environ, {"OPENEVOLVE_HANDOFF_FILE": "/nonexistent/h.json"}), \
                patch("openevolve.llm.subscription.asyncio.create_subprocess_exec", create), \
                patch("openevolve.llm.subscription._sleep_unless_shutdown", AsyncMock()):
            self.assertEqual(await init_codex(config()).generate("p"), "ok")


if __name__ == "__main__":
    unittest.main()
