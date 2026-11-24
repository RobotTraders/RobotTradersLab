from datetime import datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies.futures import (
    EquityRatio,
    Margin,
    Notional,
    Quantity,
    RiskRatio,
    TotalBalanceRatio,
)
from robottraderslab_core.engine import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    AccountProtocol,
    fetch_account_snapshots,
)
from robottraderslab_core.exchanges import (
    AccountState,
    AccountStateRequest,
    MarginMode,
    MarginSettings,
)
from robottraderslab_core.strategies import AccountRequirements, AccountSnapshot

BTCUSDT = Symbol.create("BTC/USDT:USDT")
ETHUSDT = Symbol.create("ETH/USDT:USDT")
ETHUSDC = Symbol.create("ETH/USDC:USDC")
EXECUTIONS_SINCE = datetime(2026, 5, 1)


def _account_named(name: str) -> Mock:
    account = Mock(spec=AccountProtocol)
    account.name = name
    return account


def _account_needing_a_snapshot(name: str) -> Mock:
    account = Mock(spec=AccountProtocol)
    account.name = name
    account._snapshot.return_value = Mock(spec=AccountSnapshot)
    return account


def _venue_answering_nothing() -> Mock:
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.get_account_state.return_value = AccountState(
        equities={}, conversion_rates={}
    )
    return exchange


class TestDeclarations:
    def test_each_account_keeps_its_own_declaration(self):
        requirements = AccountRequirements()

        requirements.add(_account_named("main"), positions=True)
        requirements.add(_account_named("hedge"), balances=True)

        declared = requirements._get_all()

        assert [requirement.account.name for requirement in declared] == [
            "main",
            "hedge",
        ]
        assert declared[0].positions is True
        assert declared[1].balances is True

    def test_symbols_are_captured_for_the_account(self):
        requirements = AccountRequirements()

        requirements.add(_account_named("main"), symbols=[BTCUSDT, ETHUSDT])

        assert requirements._get_all()[0].symbols == (BTCUSDT, ETHUSDT)

    def test_triggers_are_not_declared_by_default(self):
        requirements = AccountRequirements()

        requirements.add(_account_named("main"), open_orders=True)

        assert requirements._get_all()[0].triggers is False

    def test_triggers_without_the_open_orders_are_rejected(self):
        requirements = AccountRequirements()

        with pytest.raises(BotCriticalError, match="triggers.*open_orders"):
            requirements.add(_account_named("main"), triggers=True)

        assert requirements._get_all() == []

    def test_margin_settings_is_a_flag_over_those_symbols(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"), symbols=[BTCUSDT], margin_settings=True
        )

        assert requirements._get_all()[0].margin_settings is True


class TestMarginTargetDeclarations:
    def test_targets_declare_the_margin_settings_read_of_their_symbols(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"),
            symbols=[BTCUSDT],
            margin_targets={
                ETHUSDT: MarginSettings(leverage=2.0, margin_mode=MarginMode.CROSS)
            },
        )

        declared = requirements._get_all()[0]
        assert declared.margin_settings is True
        assert declared.symbols == (BTCUSDT, ETHUSDT)


class TestSizingDeclarations:
    def test_a_rule_reading_equity_declares_the_margin_currencies(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"),
            symbols=[BTCUSDT, ETHUSDT, ETHUSDC],
            sizing=[TotalBalanceRatio(0.1), EquityRatio(0.1)],
        )

        assert requirements._get_all()[0].equity == ("USDT", "USDC")

    def test_rules_reading_no_equity_declare_none(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"),
            symbols=[BTCUSDT],
            sizing=[TotalBalanceRatio(0.1), RiskRatio(0.02, of="total_balance")],
        )

        assert requirements._get_all()[0].equity == ()

    def test_a_rule_reading_a_balance_declares_the_balances(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"), symbols=[BTCUSDT], sizing=[TotalBalanceRatio(0.1)]
        )

        assert requirements._get_all()[0].balances is True

    def test_a_fixed_quantity_declares_no_balances(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"), symbols=[BTCUSDT], sizing=[Quantity(0.5)]
        )

        assert requirements._get_all()[0].balances is False

    def test_a_rule_reading_the_leverage_declares_the_margin_settings(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"),
            symbols=[BTCUSDT],
            sizing=[TotalBalanceRatio(0.1), Margin(40.0)],
        )

        assert requirements._get_all()[0].margin_settings is True

    def test_rules_reading_no_leverage_declare_no_margin_settings(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"),
            symbols=[BTCUSDT],
            sizing=[TotalBalanceRatio(0.1), Notional(200.0)],
        )

        assert requirements._get_all()[0].margin_settings is False

    def test_rules_given_as_a_generator_declare_both_reads(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"),
            symbols=[BTCUSDT],
            sizing=(rule for rule in [EquityRatio(0.1), Margin(40.0)]),
        )

        (requirement,) = requirements._get_all()
        assert requirement.margin_settings is True
        assert requirement.equity == ("USDT",)

    def test_equity_declared_beside_the_rules_is_kept_once(self):
        requirements = AccountRequirements()

        requirements.add(
            _account_named("main"),
            symbols=[BTCUSDT],
            equity=["USDC", "USDT"],
            sizing=[RiskRatio(0.02)],
        )

        assert requirements._get_all()[0].equity == ("USDC", "USDT")


class TestRequiringExecutions:
    def test_only_the_named_account_starts_reading_executions(self):
        requirements = AccountRequirements()
        requirements.add(_account_named("main"))
        requirements.add(_account_named("hedge"))

        requirements._require_executions("hedge")

        assert requirements._get_all()[0].executions is False
        assert requirements._get_all()[1].executions is True


class TestAccountNaming:
    def test_declaring_the_same_account_twice_is_rejected(self):
        account = _account_named("main")
        requirements = AccountRequirements()
        requirements.add(account, positions=True)

        with pytest.raises(BotCriticalError, match="already declared"):
            requirements.add(account, balances=True)

    def test_two_accounts_sharing_a_name_are_rejected(self):
        requirements = AccountRequirements()
        requirements.add(_account_named("bitget"), positions=True)

        with pytest.raises(BotCriticalError, match="distinct names"):
            requirements.add(_account_named("bitget"), balances=True)


class TestFetchAccountSnapshots:
    async def test_report_fills_reads_an_account_declaring_only_a_symbol(self):
        account = _account_needing_a_snapshot("main")
        requirements = AccountRequirements()
        requirements.add(account, symbols=[BTCUSDT])

        await fetch_account_snapshots(
            requirements, executions_since=EXECUTIONS_SINCE, report_fills=True
        )

        account._snapshot.assert_awaited_once_with(
            requirements._get_all()[0],
            AccountStateRequest(
                symbols=(BTCUSDT,),
                executions=True,
                max_attempts=MAX_ATTEMPTS,
                base_delay=BASE_DELAY_SECONDS,
            ),
            executions_since=EXECUTIONS_SINCE,
        )

    async def test_without_report_fills_an_account_declaring_only_a_symbol_is_skipped(
        self,
    ):
        account = _account_needing_a_snapshot("main")
        requirements = AccountRequirements()
        requirements.add(account, symbols=[BTCUSDT])

        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        account._snapshot.assert_not_awaited()

    async def test_report_fills_alone_does_not_read_an_account_with_no_symbols(self):
        account = _account_needing_a_snapshot("main")
        requirements = AccountRequirements()
        requirements.add(account)

        await fetch_account_snapshots(
            requirements, executions_since=EXECUTIONS_SINCE, report_fills=True
        )

        account._snapshot.assert_not_awaited()


class TestWidenedForExitOnly:
    async def test_an_account_declaring_an_exit_only_symbol_reads_what_exit_only_mode_judges_it_by(
        self,
    ):
        account = _account_needing_a_snapshot("main")
        requirements = AccountRequirements()
        requirements.add(account, symbols=[BTCUSDT, ETHUSDT])

        await fetch_account_snapshots(
            requirements._widened_for_exit_only(frozenset({BTCUSDT}), frozenset()),
            executions_since=EXECUTIONS_SINCE,
        )

        read_for, request = account._snapshot.await_args.args
        assert request.positions is True
        assert request.open_orders is True
        assert request.triggers is True
        assert request.symbols == (BTCUSDT, ETHUSDT)
        assert read_for.positions is True

    async def test_the_declaration_the_strategy_made_is_left_as_it_was(self):
        account = _account_needing_a_snapshot("main")
        requirements = AccountRequirements()
        requirements.add(account, symbols=[BTCUSDT])

        await fetch_account_snapshots(
            requirements._widened_for_exit_only(frozenset({BTCUSDT}), frozenset()),
            executions_since=EXECUTIONS_SINCE,
        )

        declared = requirements._get_all()[0]
        assert declared.positions is False
        assert declared.open_orders is False
        assert declared.triggers is False

    async def test_a_declaration_widened_for_exit_only_mode_keeps_what_else_it_read(
        self,
    ):
        account = _account_needing_a_snapshot("main")
        requirements = AccountRequirements()
        requirements.add(account, symbols=[BTCUSDT], balances=True)

        await fetch_account_snapshots(
            requirements._widened_for_exit_only(frozenset({BTCUSDT}), frozenset()),
            executions_since=EXECUTIONS_SINCE,
        )

        assert account._snapshot.await_args.args[1].balances is True

    async def test_an_account_declaring_no_exit_only_symbol_reads_what_it_declared(
        self,
    ):
        other = _account_needing_a_snapshot("other")
        requirements = AccountRequirements()
        requirements.add(other, symbols=[ETHUSDT])

        await fetch_account_snapshots(
            requirements._widened_for_exit_only(frozenset({BTCUSDT}), frozenset()),
            executions_since=EXECUTIONS_SINCE,
        )

        other._snapshot.assert_not_awaited()

    async def test_an_account_reading_nothing_for_an_exit_only_symbol_still_reads_its_declaration(
        self,
    ):
        other = _account_needing_a_snapshot("other")
        requirements = AccountRequirements()
        requirements.add(other, symbols=[ETHUSDT], positions=True)

        await fetch_account_snapshots(
            requirements._widened_for_exit_only(frozenset({BTCUSDT}), frozenset()),
            executions_since=EXECUTIONS_SINCE,
        )

        request = other._snapshot.await_args.args[1]
        assert request.positions is True
        assert request.open_orders is False

    def test_no_exit_only_symbol_leaves_the_declaration_itself_to_read(self):
        requirements = AccountRequirements()
        requirements.add(_account_named("main"), symbols=[BTCUSDT])

        assert (
            requirements._widened_for_exit_only(frozenset(), frozenset())
            is requirements
        )


class TestStateRequestHandedToTheConnector:
    @pytest.fixture
    def exchange(self) -> Mock:
        return _venue_answering_nothing()

    async def test_the_same_request_object_on_every_candle(self, exchange):
        requirements = AccountRequirements()
        requirements.add(FuturesAccount(exchange), symbols=[BTCUSDT], positions=True)

        for _ in range(3):
            await fetch_account_snapshots(
                requirements, executions_since=EXECUTIONS_SINCE
            )

        handed = [call.args[0] for call in exchange.get_account_state.await_args_list]
        assert len(handed) == 3
        assert handed[1] is handed[0]
        assert handed[2] is handed[0]

    @pytest.mark.parametrize("triggers", [False, True])
    async def test_the_request_carries_the_triggers_the_strategy_declared(
        self, exchange, triggers
    ):
        requirements = AccountRequirements()
        requirements.add(
            FuturesAccount(exchange),
            symbols=[BTCUSDT],
            open_orders=True,
            triggers=triggers,
        )

        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        (call,) = exchange.get_account_state.await_args_list
        assert call.args[0].triggers is triggers

    async def test_an_account_declared_after_a_read_is_read_on_the_next_candle(
        self, exchange
    ):
        hedge_exchange = _venue_answering_nothing()
        requirements = AccountRequirements()
        requirements.add(FuturesAccount(exchange, name="main"), positions=True)
        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        requirements.add(
            FuturesAccount(hedge_exchange, name="hedge"),
            symbols=[BTCUSDT],
            balances=True,
        )
        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        hedge_request = hedge_exchange.get_account_state.await_args.args[0]
        assert hedge_request.balances is True

    async def test_a_tracker_position_need_after_a_read_is_in_the_next_request(
        self, exchange
    ):
        requirements = AccountRequirements()
        requirements.add(FuturesAccount(exchange, name="main"), balances=True)
        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        requirements._require_positions("main", (BTCUSDT,))
        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        after = exchange.get_account_state.await_args.args[0]
        assert after.positions is True
        assert after.symbols == (BTCUSDT,)

    async def test_a_tracker_execution_need_after_a_read_is_in_the_next_request(
        self, exchange
    ):
        requirements = AccountRequirements()
        requirements.add(FuturesAccount(exchange, name="main"), balances=True)
        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        requirements._require_executions("main")
        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        after = exchange.get_account_state.await_args.args[0]
        assert after.executions is True

    async def test_a_run_without_fill_reports_reads_no_executions_after_one_with(
        self, exchange
    ):
        requirements = AccountRequirements()
        requirements.add(FuturesAccount(exchange), symbols=[BTCUSDT], positions=True)
        await fetch_account_snapshots(
            requirements, executions_since=EXECUTIONS_SINCE, report_fills=True
        )

        await fetch_account_snapshots(requirements, executions_since=EXECUTIONS_SINCE)

        last_request = exchange.get_account_state.await_args.args[0]
        assert last_request.executions is False
