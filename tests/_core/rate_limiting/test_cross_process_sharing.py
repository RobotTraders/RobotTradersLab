import os
import subprocess
import sys
from pathlib import Path

import pytest

CAP_PER_SECOND = 5
SENDS_PER_PROCESS = 8
CHILD_TIMEOUT_SECONDS = 60
# The limiter caps grants per rolling second of its own clock, and a child prints its timestamp after `wait` returns, so preemption on
# a busy CI runner lands between the grant and the timestamp.
# The assertion window leaves that skew 0.4s of budget while still
# catching a limiter that overshoots its cap.
WINDOW_LESS_SEND_JITTER_SECONDS = 0.6
NO_STALL_SECONDS = 0.0
MID_RUN_STALL_SECONDS = 0.4

_CHILD = """
import asyncio, sys, time
from pathlib import Path
from robottraderslab._core.rate_limiting import SharedRequestWindow

async def main():
    scope, gate, sends, cap, stall = (
        sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        float(sys.argv[5]),
    )
    window = SharedRequestWindow(scope, max_per_second={"budget": cap})
    while not gate.exists():
        await asyncio.sleep(0.005)
    for index in range(sends):
        await window.wait("budget")
        print(time.monotonic(), flush=True)
        if stall and index == sends // 2:
            time.sleep(stall)

asyncio.run(main())
"""


def test_processes_on_one_scope_stay_under_the_cap_together(tmp_path):
    gate = tmp_path / "gate"

    smooth = _start_child(gate, tmp_path, stall_seconds=NO_STALL_SECONDS)
    stalling = _start_child(gate, tmp_path, stall_seconds=MID_RUN_STALL_SECONDS)
    gate.touch()
    send_times = sorted(_send_times(smooth) + _send_times(stalling))

    assert len(send_times) == 2 * SENDS_PER_PROCESS
    assert _rolling_window_peak(send_times) <= CAP_PER_SECOND


def _rolling_window_peak(send_times: list[float]) -> int:
    peak = 0
    window_start = 0
    for index in range(len(send_times)):
        while (
            send_times[index] - send_times[window_start]
            > WINDOW_LESS_SEND_JITTER_SECONDS
        ):
            window_start += 1
        peak = max(peak, index - window_start + 1)
    return peak


def _start_child(
    gate: Path, state_dir: Path, *, stall_seconds: float
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD,
            "cross-process-scope",
            str(gate),
            str(SENDS_PER_PROCESS),
            str(CAP_PER_SECOND),
            str(stall_seconds),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "TMP": str(state_dir),
            "TEMP": str(state_dir),
            "TMPDIR": str(state_dir),
        },
    )


def _send_times(child: subprocess.Popen[str]) -> list[float]:
    stdout, stderr = child.communicate(timeout=CHILD_TIMEOUT_SECONDS)
    if child.returncode != 0:
        pytest.fail(f"child exited with {child.returncode}: {stderr}")
    return [float(line) for line in stdout.split()]


ANNOUNCED_COUNT = 3
ANNOUNCED_RESET_SECONDS = 1.0
CLAIMS_PER_PROCESS = 3
RESET_LESS_START_SKEW_SECONDS = 0.85

_VENUE_CHILD = """
import asyncio, sys, time
from pathlib import Path
from robottraderslab._core.rate_limiting import SharedVenueWindow

async def main():
    scope, gate, claims, count, reset, stall = (
        sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
        float(sys.argv[5]), float(sys.argv[6]),
    )
    window = SharedVenueWindow(scope)
    while not gate.exists():
        await asyncio.sleep(0.005)
    await window.record("bucket", remaining=count, window="w1", reset_after=reset)
    for index in range(claims):
        await window.wait("bucket")
        print(time.monotonic(), flush=True)
        if stall and index == claims // 2:
            time.sleep(stall)

asyncio.run(main())
"""


def test_processes_on_one_announced_count_wait_for_its_reset_together(tmp_path):
    gate = tmp_path / "gate"

    smooth = _start_venue_child(gate, tmp_path, stall_seconds=NO_STALL_SECONDS)
    stalling = _start_venue_child(gate, tmp_path, stall_seconds=MID_RUN_STALL_SECONDS)
    gate.touch()
    claim_times = sorted(_send_times(smooth) + _send_times(stalling))

    assert len(claim_times) == 2 * CLAIMS_PER_PROCESS
    assert (
        claim_times[ANNOUNCED_COUNT] - claim_times[0] >= RESET_LESS_START_SKEW_SECONDS
    )


def _start_venue_child(
    gate: Path, state_dir: Path, *, stall_seconds: float
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            _VENUE_CHILD,
            "cross-process-venue-scope",
            str(gate),
            str(CLAIMS_PER_PROCESS),
            str(ANNOUNCED_COUNT),
            str(ANNOUNCED_RESET_SECONDS),
            str(stall_seconds),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "TMP": str(state_dir),
            "TEMP": str(state_dir),
            "TMPDIR": str(state_dir),
        },
    )
