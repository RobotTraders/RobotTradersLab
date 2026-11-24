from typing import Protocol

from robottraderslab_core import TimeFrame


class ProfileProtocol(Protocol):
    """What `ProfileStrategy` reads off a profile, which `Profile` provides:
    an id unique among the strategy's profiles and the timeframe whose candle
    close runs it.
    """

    @property
    def profile_id(self) -> str: ...

    @property
    def timeframe(self) -> TimeFrame: ...
