import json
from dataclasses import replace
from typing import Any

import pytest

from robottraderslab.chart.encoding import encode_payload
from robottraderslab.chart.payload import ChartPayload, TradeRow


@pytest.fixture
def encoded(payload: ChartPayload) -> dict[str, Any]:
    return json.loads(encode_payload(payload))


class TestEncodePayload:
    def test_candles_travel_as_positional_arrays(self, encoded):
        assert encoded["markets"][0]["candles"] == [[1735689600, 1.0, 4.0, 0.5, 3.0]]

    def test_points_travel_as_positional_arrays(self, encoded):
        assert encoded["equity"] == [[1735689600, 1000.0]]
        assert encoded["markets"][0]["series"][0]["points"] == [[1735689600, 2.5]]

    def test_enums_encode_as_their_string_values(self, encoded):
        market = encoded["markets"][0]
        assert market["series"][0]["pane"] == "separate"
        assert market["series"][0]["shape"] == "line"
        assert market["markers"][0]["side"] == "buy"

    def test_a_marker_carries_its_time_side_and_label(self, encoded):
        assert encoded["markets"][0]["markers"] == [
            {"time": 1735689600, "side": "buy", "label": "long"}
        ]

    def test_a_market_carries_its_label_and_settings(self, encoded):
        market = encoded["markets"][0]
        assert market["label"] == "BTC/USDT:USDT · 1h"
        assert market["settings"] == [{"name": "Trix length", "value": "31"}]

    def test_links_carry_their_target_and_state(self, encoded):
        assert encoded["links"][0]["href"] == "BTC-USDT-USDT_1h_alpha.html"
        assert encoded["links"][0]["current"] is True
        assert encoded["links"][1]["current"] is False

    def test_links_carry_their_profiles_settings(self, encoded):
        assert encoded["links"][0]["settings"] == [
            {"name": "Trix length", "value": "31"}
        ]

    def test_trades_travel_as_keyed_objects(self, encoded):
        trades = encoded["markets"][0]["trades"]
        assert trades[0]["side"] == "long"
        assert trades[0]["net_pnl_pct"] == 200.0

    def test_an_open_trade_encodes_a_null_exit(self, payload, market):
        still_open = TradeRow(
            entry_time=1735689600,
            exit_time=None,
            side="long",
            entry_price=1.0,
            exit_price=None,
            net_pnl=0.0,
            net_pnl_pct=0.0,
            entry_reason="ma_cross",
            exit_reason=None,
        )

        open_market = replace(market, trades=[still_open])
        encoded = json.loads(encode_payload(replace(payload, markets=[open_market])))

        trade = encoded["markets"][0]["trades"][0]
        assert trade["exit_time"] is None
        assert trade["exit_price"] is None
        assert trade["exit_reason"] is None

    def test_the_headline_travels_beside_the_sections(self, encoded):
        assert encoded["report"]["headline"] == [
            {"label": "Performance", "value": "9.50%"}
        ]

    def test_report_sections_keep_their_titles(self, encoded):
        assert encoded["report"]["sections"][0]["title"] == "Returns"

    def test_report_figures_travel_as_the_strings_they_will_be_shown_as(self, encoded):
        assert encoded["report"]["sections"][0]["rows"] == [
            {"label": "Return", "value": "9.50%"},
            {"label": "Profit factor", "value": "∞"},
        ]

    def test_the_trades_table_travels_with_its_columns(self, encoded):
        assert encoded["report"]["trades"]["title"] == "Trades"
        assert encoded["report"]["trades"]["columns"] == ["All", "Long"]

    def test_a_table_row_carries_one_value_per_column(self, encoded):
        groups = encoded["report"]["trades"]["groups"]
        assert groups[0]["rows"] == [{"label": "Total trades", "values": ["4", "2"]}]

    def test_an_untitled_group_travels_with_a_null_title(self, encoded):
        groups = encoded["report"]["trades"]["groups"]
        assert groups[0]["title"] is None
        assert groups[1]["title"] == "Fees"

    def test_the_configuration_text_travels_with_the_payload(self, encoded):
        assert encoded["configuration"] == '[strategy]\nstrategy_class = "impulse"\n'

    def test_output_carries_no_insignificant_whitespace(self, payload):
        document = encode_payload(payload)

        assert ", " not in document
        assert "\n" not in document

    def test_angle_brackets_cannot_close_the_host_script_block(self, payload):
        markup = "</script><script>alert(1)</script>"

        document = encode_payload(replace(payload, title=markup))

        assert "<" not in document
        assert json.loads(document)["title"] == markup
