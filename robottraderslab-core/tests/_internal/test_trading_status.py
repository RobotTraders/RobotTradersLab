from robottraderslab_core.exchanges import Closure, TradingStatus

STATUSES = {
    "limit_open": TradingStatus.CLOSED_TO_NEW_POSITIONS,
    "maintain": TradingStatus.CLOSED_TO_EVERY_ORDER,
}


class TestClosureFromWords:
    def test_the_first_word_the_statuses_name_closes_the_symbol(self):
        closure = Closure.from_words(STATUSES, ["normal", "limit_open", "maintain"])

        assert closure == Closure(TradingStatus.CLOSED_TO_NEW_POSITIONS, "limit_open")

    def test_no_word_the_statuses_name_leaves_the_symbol_open(self):
        closure = Closure.from_words(STATUSES, ["normal", "listed"])

        assert closure is None
