"""Usage-limit handling for subscription-backed CLIs.

A subscription is not an API key: it runs out. When the provider says the quota
window is spent, the evolution must neither burn its remaining iterations on
instant failures nor record a spurious zero-score child. Instead:

* a reset at most ``OPENEVOLVE_LIMIT_WAIT_MAX_SECONDS`` away (default 1 h) is
  waited out inside the call, which then retries the same prompt -- the
  iteration never sees the limit;
* a reset further out is *handed off*: the adapter writes the JSON file named
  by ``OPENEVOLVE_HANDOFF_FILE`` and raises :class:`UsageLimitHandoff`. The
  parent loop (process_parallel) sees the file, discards the affected
  iteration, checkpoints and stops, and the caller (e.g. a SLURM job script)
  resumes from that checkpoint once ``resume_at`` has passed.

Without ``OPENEVOLVE_HANDOFF_FILE`` every limit is waited out in place, in
sleeps of at most ``OPENEVOLVE_LIMIT_MAX_SLEEP_SECONDS`` (default 6 h).

The detection and reset parsing mirror the autoresearch agent_driver: only
error-bearing text is matched (CLI error events and stderr ``ERROR:`` lines),
never the model's own answer.
"""

import datetime
import json
import os
import re
import time
from typing import Iterable, Optional

LIMIT_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"usage[ _-]?limit",
        r"rate[ _-]?limit",
        r"hit your (session|usage) limit",
        r"limit will reset",
        r"quota",
        r"resource.*exhausted",
        r"at capacity",
        r"(api error|http|status(?: code)?)[^0-9]{0,16}429\b",
    )
]


# Failures that never clear on their own (a dead login, a broken sandbox).
# Retrying them only turns every remaining iteration into a quick failure, so
# the run is stopped instead and not resubmitted. Mirrors fatal_patterns in
# the autoresearch driver.yaml (codex + claude adapters).
FATAL_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"access token could not be refreshed",
        r"could not be refreshed",
        r"failed to authenticate",
        r"oauth session expired",
        r"not logged in",
        r"bwrap: Creating new namespace failed",
        r"fs sandbox helper failed(?: with status|:)",
    )
]


class SubscriptionFatal(RuntimeError):
    """A CLI failure that retrying cannot fix; the run must stop for a human."""


class UsageLimitHit(RuntimeError):
    """The provider rejected a call because a quota window is spent."""

    def __init__(self, message: str, reset_epoch: Optional[int]):
        super().__init__(message)
        self.reset_epoch = reset_epoch


class UsageLimitHandoff(RuntimeError):
    """A limit too distant to wait out; a handoff file has been written."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def codex_diagnostics(stdout: str, stderr: str) -> str:
    """Error-bearing text from a ``codex exec --json`` run.

    Error events (``{"type": "error"}``, ``{"type": "turn.failed"}``) and
    stderr ``ERROR:`` lines, minus in-CLI reconnect progress lines that the
    CLI recovered from on its own.
    """
    parts = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") in ("error", "turn.failed"):
            parts.append(json.dumps(event))
    for line in stderr.splitlines():
        if line.startswith("ERROR:") and not line.startswith("ERROR: Reconnecting..."):
            parts.append(line)
    return "\n".join(parts)


def match_fatal(text: str) -> bool:
    return any(p.search(text) for p in FATAL_PATTERNS)


def match_limit(diagnostics: str) -> bool:
    return any(p.search(diagnostics) for p in LIMIT_PATTERNS)


def parse_reset_epoch(blob: str, now: Optional[float] = None) -> Optional[int]:
    """Best-effort epoch at which the spent window reopens, or None."""
    now = time.time() if now is None else now

    epochs = [
        int(e)
        for e in re.findall(r'"resets_?[Aa]t"\s*:\s*"?(\d{10})', blob)
    ]
    future = [e for e in epochs if e > now]
    if future:
        # Earliest: waking too early costs one retry, too late costs the run.
        return min(future)

    m = re.search(r"limit[^|\n]{0,40}\|\s*(\d{10})", blob, re.I)
    if m:
        return int(m.group(1))

    # Codex prose, rendered in the node's local time:
    #   "... or try again at Sep 15th, 2026 3:24 AM."
    m = re.search(
        r"try again (?:at|on)\s+"
        r"([A-Z][a-z]{2})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})"
        r"[,\s]+(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]",
        blob,
    )
    if m:
        months = ("jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec")
        name, day, year, hour, minute, meridiem = m.groups()
        try:
            month = months.index(name.lower()) + 1
            hour24 = int(hour) % 12 + (12 if meridiem.lower() == "p" else 0)
            return int(datetime.datetime(int(year), month, int(day),
                                         hour24, int(minute)).timestamp())
        except ValueError:
            pass

    # "Resets in 35m39s" / "try again in 2h 5m"
    m = None
    for m in re.finditer(
        r"(?:[Rr]esets|try again) in\s+((?:\d+\s*h\s*)?(?:\d+\s*m\s*)?(?:\d+(?:\.\d+)?\s*s)?)",
        blob,
    ):
        pass
    if m and m.group(1).strip():
        parts = {u: float(v) for v, u in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", m.group(1))}
        return int(now + parts.get("h", 0) * 3600 + parts.get("m", 0) * 60 + parts.get("s", 0))

    return None


def write_handoff(path: str, *, resume_at: float, message: str, reason: str) -> None:
    """Atomically record that the run must stop and resume at ``resume_at``.

    The first writer wins: a later limit (e.g. a queued iteration hitting the
    same window) must not move an already-recorded resume time.
    """
    if os.path.exists(path):
        return
    record = {
        "reason": reason,
        "resume_at": int(resume_at),
        "resume_at_iso": datetime.datetime.fromtimestamp(int(resume_at)).isoformat(),
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "message": message[-4000:],
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)


# Handoff-file reasons after which the run must NOT be resumed automatically.
FATAL_REASONS = ("fatal", "llm_failures")


def handoff_requested() -> bool:
    path = os.environ.get("OPENEVOLVE_HANDOFF_FILE")
    return bool(path) and os.path.exists(path)


def plan_limit_response(hit: UsageLimitHit, consecutive_waits: int,
                        now: Optional[float] = None) -> tuple:
    """Decide how to react to a limit: ("wait", seconds) or ("handoff", resume_at).

    Raises RuntimeError when no handoff is configured and the run has waited
    ``OPENEVOLVE_MAX_LIMIT_WAITS`` times in a row without the window reopening.
    """
    now = time.time() if now is None else now
    margin = _env_float("OPENEVOLVE_LIMIT_RESET_MARGIN_SECONDS", 120)
    poll = _env_float("OPENEVOLVE_LIMIT_POLL_SECONDS", 900)
    wait_max = _env_float("OPENEVOLVE_LIMIT_WAIT_MAX_SECONDS", 3600)
    max_sleep = _env_float("OPENEVOLVE_LIMIT_MAX_SLEEP_SECONDS", 21600)
    max_waits = int(_env_float("OPENEVOLVE_MAX_LIMIT_WAITS", 12))
    can_handoff = bool(os.environ.get("OPENEVOLVE_HANDOFF_FILE"))

    known = hit.reset_epoch is not None and hit.reset_epoch > now
    delay = (hit.reset_epoch - now + margin) if known else poll

    if can_handoff:
        if known and delay > wait_max:
            return "handoff", now + delay
        if consecutive_waits > max_waits:
            # An unparseable limit that keeps coming back: stop holding the
            # allocation and try again in an hour.
            return "handoff", now + max(wait_max, poll)
        return "wait", delay

    if consecutive_waits > max_waits:
        raise RuntimeError(
            f"usage limit persisted through {max_waits} waits: {hit}")
    return "wait", min(delay, max_sleep)
