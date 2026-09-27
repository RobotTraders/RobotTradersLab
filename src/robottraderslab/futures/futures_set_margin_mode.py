from dataclasses import dataclass

from robottraderslab._core import ActionResult, BaseExchangeAction, MarginMode, Symbol

from .futures_exchange_protocol import FuturesExchangeProtocol


@dataclass(kw_only=True, slots=True)
class SetMarginModeAction(BaseExchangeAction):
    """Sets how margin backs the account's positions on the symbol."""

    exchange: FuturesExchangeProtocol
    symbol: Symbol
    margin_mode: MarginMode

    async def execute(self) -> ActionResult:
        await self.exchange.set_margin_mode(self.symbol, self.margin_mode)
        return ActionResult()
