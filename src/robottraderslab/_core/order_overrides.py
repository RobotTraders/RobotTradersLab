from dataclasses import dataclass, replace

from .order import (
    NO_VENUE_PARAMS,
    StopLoss,
    TakeProfit,
    TimeInForce,
    TriggerType,
    VenueParams,
)


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderSettings:
    """What a run's configuration sets on every order of one role, over what
    the strategy's code sets for it.

    Attributes:
        time_in_force: Taken by each limit order of the role that waits on no
            trigger, since a triggered order rests until cancelled; None
            leaves each order the strategy's.
        trigger_type: Taken by each order of the role that waits on a
            trigger; None leaves each order the strategy's.
        params: Venue fields, each taking the place of the strategy's field
            of the same name for the same venue.
    """

    time_in_force: TimeInForce | None = None
    trigger_type: TriggerType | None = None
    params: VenueParams = NO_VENUE_PARAMS

    def params_over(self, strategy_params: VenueParams) -> VenueParams:
        if not self.params:
            return strategy_params
        return {
            venue: {**strategy_params.get(venue, {}), **self.params.get(venue, {})}
            for venue in strategy_params.keys() | self.params.keys()
        }

    def time_in_force_over(self, strategy_value: TimeInForce) -> TimeInForce:
        return strategy_value if self.time_in_force is None else self.time_in_force

    def trigger_type_over(self, strategy_value: TriggerType) -> TriggerType:
        return strategy_value if self.trigger_type is None else self.trigger_type


NO_ORDER_SETTINGS = OrderSettings()


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderOverrides:
    """The order settings a run's configuration sets over the strategy's code.

    Attributes:
        entry: Every order that opens or grows a position.
        exit: Every order that shrinks or closes one.
        stop_loss_trigger_type: Taken by every stop-loss, attached to an
            order or moved on a position; None leaves each the strategy's.
        take_profit_trigger_type: Taken by every take-profit the same way.
    """

    entry: OrderSettings = NO_ORDER_SETTINGS
    exit: OrderSettings = NO_ORDER_SETTINGS
    stop_loss_trigger_type: TriggerType | None = None
    take_profit_trigger_type: TriggerType | None = None

    def stop_loss_over(self, stop_loss: StopLoss | None) -> StopLoss | None:
        if stop_loss is None or self.stop_loss_trigger_type is None:
            return stop_loss
        return replace(stop_loss, trigger_type=self.stop_loss_trigger_type)

    def stop_loss_trigger_type_over(self, strategy_value: TriggerType) -> TriggerType:
        configured = self.stop_loss_trigger_type
        return strategy_value if configured is None else configured

    def take_profit_over(self, take_profit: TakeProfit | None) -> TakeProfit | None:
        if take_profit is None or self.take_profit_trigger_type is None:
            return take_profit
        return replace(take_profit, trigger_type=self.take_profit_trigger_type)

    def take_profit_trigger_type_over(self, strategy_value: TriggerType) -> TriggerType:
        configured = self.take_profit_trigger_type
        return strategy_value if configured is None else configured


NO_ORDER_OVERRIDES = OrderOverrides()
