import asyncio
import logging
import time

from robottraderslab.exchanges import SharedRequestWindow

SCOPE = "emission-scope"
KEY = "budget"
CAP = 8
REQUESTS = 30
CONCURRENT_REQUESTS = 10
SIMULATED_RTT_SECONDS = 0.05
STALL_SECONDS = 0.25
STALLS = 4
BETWEEN_STALLS_SECONDS = 0.5
# The limiter caps grants per rolling second of its own clock, and a send is timestamped after `wait` returns, so a deliberate loop
# stall (0.25s) or a busy CI runner lands between the grant and the
# timestamp.
# The assertion window leaves that skew 0.4s of budget while still
# catching a limiter that overshoots its cap.
WINDOW_LESS_SEND_JITTER_SECONDS = 0.6


def _rolling_window_peak(send_times: list[float]) -> int:
    ordered = sorted(send_times)
    peak = 0
    window_start = 0
    for index in range(len(ordered)):
        while ordered[index] - ordered[window_start] > WINDOW_LESS_SEND_JITTER_SECONDS:
            window_start += 1
        peak = max(peak, index - window_start + 1)
    return peak


async def _paced_sends(window: SharedRequestWindow) -> list[float]:
    send_times: list[float] = []
    slots = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def one_request() -> None:
        async with slots:
            await window.wait(KEY)
            send_times.append(time.monotonic())
            await asyncio.sleep(SIMULATED_RTT_SECONDS)

    await asyncio.gather(*(one_request() for _ in range(REQUESTS)))
    return send_times


async def test_emissions_stay_under_the_cap_while_the_loop_stalls(caplog):
    window = SharedRequestWindow(SCOPE, max_per_second={KEY: CAP})
    stalling = asyncio.create_task(_stall_the_loop())

    with caplog.at_level(logging.WARNING):
        send_times = await _paced_sends(window)
    stalling.cancel()

    assert "not shared between processes" not in caplog.text
    assert len(send_times) == REQUESTS
    assert _rolling_window_peak(send_times) <= CAP


async def _stall_the_loop() -> None:
    for _ in range(STALLS):
        await asyncio.sleep(BETWEEN_STALLS_SECONDS)
        time.sleep(STALL_SECONDS)
