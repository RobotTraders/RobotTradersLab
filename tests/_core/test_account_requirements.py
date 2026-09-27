from datetime import datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    AccountProtocol,
    AccountRequirements,
    AccountSnapshot,
    MarginMode,
    MarginSettings,
    fetch_account_snapshots,
)
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies.futures import (
    EquityRatio,
    Margin,
    Notional,
    Quantity,
    RiskRatio,
    TotalBalanceRatio,
)

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

        with pytest.raises(StrategyCriticalError, match="already declared"):
            requirements.add(account, balances=True)

    def test_two_accounts_sharing_a_name_are_rejected(self):
        requirements = AccountRequirements()
        requirements.add(_account_named("bitget"), positions=True)

        with pytest.raises(StrategyCriticalError, match="distinct names"):
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
            executions_since=EXECUTIONS_SINCE,
            report_fills=True,
            max_attempts=MAX_ATTEMPTS,
            base_delay=BASE_DELAY_SECONDS,
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
