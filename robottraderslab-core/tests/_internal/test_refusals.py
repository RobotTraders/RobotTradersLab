import logging

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core._internal import report_refusal
from robottraderslab_core._internal.refusals import collecting_margin_refusals
from robottraderslab_core.exceptions import (
    ExchangeRecoverableError,
    InsufficientMarginError,
    NoOpenPositionError,
)

ETH = Symbol.create("ETH/USDT:USDT")
HYPE = Symbol.create("HYPE/USDT:USDT")


def refuse(
    symbol: Symbol,
    error: ExchangeRecoverableError,
    *,
    profile_tag: str | None = "4h-fast",
    is_exit: bool = False,
) -> None:
    report_refusal(
        symbol, profile_tag, error, is_exit=is_exit, what="FuturesMarketOrderAction"
    )


def levels_and_messages(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    return [(record.levelno, record.getMessage()) for record in caplog.records]


class TestAnExitTheVenueRefuses:
    def test_is_an_error_naming_the_symbol_the_profile_and_the_answer(self, caplog):
        with caplog.at_level(logging.INFO):
            refuse(
                ETH,
                ExchangeRecoverableError("25202 Insufficient balance"),
                is_exit=True,
            )

        assert levels_and_messages(caplog) == [
            (
                logging.ERROR,
                "Exit refused, the strategy's position is not closed: ETH/USDT:USDT, "
                "profile ETH/USDT:USDT@4h-fast, FuturesMarketOrderAction: "
                "25202 Insufficient balance",
            )
        ]

    def test_stays_an_error_when_the_answer_is_a_margin_refusal(self, caplog):
        with caplog.at_level(logging.INFO), collecting_margin_refusals():
            refuse(ETH, InsufficientMarginError("25202"), is_exit=True)

        assert [level for level, _ in levels_and_messages(caplog)] == [logging.ERROR]

    def test_without_a_tag_says_no_profile_owns_the_order(self, caplog):
        with caplog.at_level(logging.INFO):
            refuse(
                ETH, ExchangeRecoverableError("refused"), profile_tag=None, is_exit=True
            )

        assert "no profile, the order carries no tag" in caplog.text

    def test_on_a_position_already_gone_is_an_info_that_nothing_is_left_to_close(
        self, caplog
    ):
        with caplog.at_level(logging.INFO):
            refuse(ETH, NoOpenPositionError("No open position"), is_exit=True)

        assert levels_and_messages(caplog) == [
            (
                logging.INFO,
                "FuturesMarketOrderAction not placed, the account holds no position "
                "on ETH/USDT:USDT (profile ETH/USDT:USDT@4h-fast): No open position",
            )
        ]


class TestAnEntryTheVenueRefuses:
    def test_other_than_for_margin_is_a_warning_with_the_executors_words(self, caplog):
        with caplog.at_level(logging.INFO):
            refuse(ETH, ExchangeRecoverableError("Order quantity under the step"))

        assert levels_and_messages(caplog) == [
            (
                logging.WARNING,
                "Exchange rejected FuturesMarketOrderAction (ETH/USDT:USDT): "
                "Order quantity under the step",
            )
        ]

    def test_for_margin_joins_the_one_line_of_its_cycle(self, caplog):
        with caplog.at_level(logging.INFO):
            with collecting_margin_refusals():
                refuse(ETH, InsufficientMarginError("25202 Insufficient balance"))
                refuse(ETH, InsufficientMarginError("25202 Insufficient balance"))
                refuse(HYPE, InsufficientMarginError("25203 Insufficient margin"))

        assert levels_and_messages(caplog) == [
            (
                logging.WARNING,
                "The account is fully committed, its free margin carries no more "
                "orders: 3 not placed (ETH/USDT:USDT@4h-fast x2, "
                "HYPE/USDT:USDT@4h-fast), the venue answered: "
                "25202 Insufficient balance; 25203 Insufficient margin",
            )
        ]

    def test_for_margin_with_no_profile_names_the_symbol(self, caplog):
        with caplog.at_level(logging.INFO), collecting_margin_refusals():
            refuse(ETH, InsufficientMarginError("25202"), profile_tag=None)

        assert "(ETH/USDT:USDT)," in caplog.text

    def test_names_ten_profiles_then_counts_the_rest(self, caplog):
        with caplog.at_level(logging.INFO), collecting_margin_refusals():
            for number in range(12):
                refuse(
                    ETH, InsufficientMarginError("refused"), profile_tag=f"p{number}"
                )

        assert caplog.messages[0].count("ETH/USDT:USDT@p") == 10
        assert "and 2 more" in caplog.messages[0]

    def test_quotes_three_distinct_answers_at_most(self, caplog):
        with caplog.at_level(logging.INFO), collecting_margin_refusals():
            for number in range(5):
                refuse(ETH, InsufficientMarginError(f"answer {number}"))

        assert caplog.messages[0].endswith("answer 0; answer 1; answer 2")


class TestTheLineOfACycle:
    def test_is_not_written_when_nothing_was_refused_for_margin(self, caplog):
        with caplog.at_level(logging.INFO), collecting_margin_refusals():
            pass

        assert caplog.records == []

    def test_is_written_when_the_block_raises(self, caplog):
        def refuse_then_fail() -> None:
            with collecting_margin_refusals():
                refuse(ETH, InsufficientMarginError("25202"))
                raise RuntimeError("a critical")

        with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
            refuse_then_fail()

        assert [level for level, _ in levels_and_messages(caplog)] == [logging.WARNING]

    def test_holds_nothing_for_the_next_block(self, caplog):
        with collecting_margin_refusals():
            refuse(ETH, InsufficientMarginError("first cycle"))
        caplog.clear()

        with caplog.at_level(logging.INFO), collecting_margin_refusals():
            refuse(HYPE, InsufficientMarginError("second cycle"))

        assert "ETH/USDT:USDT" not in caplog.text
        assert "second cycle" in caplog.text

    def test_is_closed_for_a_refusal_after_the_block(self, caplog):
        with collecting_margin_refusals():
            pass

        with caplog.at_level(logging.INFO):
            refuse(ETH, InsufficientMarginError("25202"))

        assert [level for level, _ in levels_and_messages(caplog)] == [logging.WARNING]
        assert "Exchange rejected" in caplog.text
