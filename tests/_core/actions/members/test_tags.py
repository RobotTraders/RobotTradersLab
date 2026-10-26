import pytest

from robottraderslab._core.actions.members.tags import profile_tag
from robottraderslab.exceptions import BotCriticalError


class TestProfileTag:
    def test_composes_timeframe_and_tag_with_a_dash(self):
        assert profile_tag("1h", "alpha") == "1h-alpha"

    def test_empty_tag_leaves_just_the_timeframe(self):
        assert profile_tag("1h", "") == "1h"

    def test_a_tag_at_the_budget_is_composed(self):
        assert profile_tag("1h", "abcdefgh") == "1h-abcdefgh"

    def test_a_tag_over_the_budget_is_refused(self):
        with pytest.raises(BotCriticalError, match="8-byte budget"):
            profile_tag("1h", "abcdefghi")

    def test_a_multi_byte_tag_counts_by_its_encoded_width(self):
        with pytest.raises(BotCriticalError, match="8-byte budget"):
            profile_tag("1h", "€€€")

    def test_a_tag_carrying_the_separator_is_refused(self):
        with pytest.raises(BotCriticalError, match="separates"):
            profile_tag("1h", "too-long")

    @pytest.mark.parametrize("tag", ["sl", "tp"])
    def test_sl_and_tp_are_refused_as_a_tag(self, tag):
        with pytest.raises(BotCriticalError, match=f"'{tag}'.*choose another"):
            profile_tag("4h", tag)

    @pytest.mark.parametrize("tag", ["slow", "xsl", "s", "SL"])
    def test_a_tag_only_resembling_a_bracket_suffix_is_composed(self, tag):
        assert profile_tag("4h", tag) == f"4h-{tag}"

    def test_an_unknown_timeframe_is_refused(self):
        with pytest.raises(BotCriticalError, match="not a timeframe"):
            profile_tag("7h", "alpha")  # type: ignore[arg-type]
