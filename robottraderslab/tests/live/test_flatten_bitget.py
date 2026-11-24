from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.live.flatten import main as flatten
from robottraderslab_core.exchanges import HttpResponse, MarginMode

BitgetClient = pytest.importorskip("robottraderslab_bitget.bitget_client").BitgetClient
BitgetHTTPError = pytest.importorskip(
    "robottraderslab_bitget.bitget_exceptions"
).BitgetHTTPError
BitgetExchange = pytest.importorskip(
    "robottraderslab_bitget.bitget_exchange"
).BitgetExchange

BTC_USDT = Symbol.create("BTC/USDT:USDT")
POSITIONS_ENDPOINT = "/api/v3/position/current-position"
ASSETS_ENDPOINT = "/api/v3/account/assets"
PLACE_ORDER_ENDPOINT = "/api/v3/trade/place-order"
FILLS_ENDPOINT = "/api/v3/trade/fills"
HISTORY_ORDERS_ENDPOINT = "/api/v3/trade/history-orders"
UNFILLED_ORDERS_ENDPOINT = "/api/v3/trade/unfilled-orders"
INSTRUMENTS_ENDPOINT = "/api/v3/market/instruments"
SETTINGS_ENDPOINT = "/api/v3/account/settings"
NO_POSITION_TO_CLOSE = "25227"

INSTRUMENTS_RESPONSE = {
    "code": "00000",
    "msg": "success",
    "requestTime": 1770531248742,
    "data": [
        {
            "symbol": "BTCUSDT",
            "category": "USDT-FUTURES",
            "baseCoin": "BTC",
            "quoteCoin": "USDT",
            "feeRateUpRatio": "0.005",
            "makerFeeRate": "0.0002",
            "takerFeeRate": "0.0006",
            "minOrderAmount": "5",
            "maxLeverage": "125",
            "openCostUpRatio": "0.01",
            "minOrderQty": "0.0001",
            "maxOrderQty": "1234",
            "maxMarketOrderQty": "567",
            "pricePrecision": "1",
            "quantityPrecision": "4",
            "quotePrecision": "8",
            "priceMultiplier": "0.1",
            "quantityMultiplier": "0.0001",
            "type": "perpetual",
            "status": "online",
        },
    ],
}

ACCOUNT_SETTINGS_RESPONSE = {
    "code": "00000",
    "msg": "success",
    "requestTime": 1753787749280,
    "data": {
        "uid": "1111111111",
        "accountMode": "hybrid",
        "assetMode": "multi_assets",
        "accountLevel": "basic",
        "holdMode": "one_way_mode",
        "stpMode": "none",
        "deltaSwitch": "no",
        "repayMode": "manual",
        "symbolConfigList": [
            {
                "category": "USDT-FUTURES",
                "symbol": "BTCUSDT",
                "marginMode": "crossed",
                "leverage": "1",
            }
        ],
        "coinConfigList": [],
    },
}

ASSETS_RESPONSE = {
    "code": "00000",
    "msg": "success",
    "requestTime": 1746687063471,
    "data": {
        "accountEquity": "99.9",
        "usdtEquity": "99.9",
        "unrealisedPnl": "0",
        "effEquity": "88.8",
        "assets": [
            {
                "coin": "USDT",
                "equity": "7.75",
                "usdValue": "6.6",
                "balance": "6.25",
                "balanceOriginal": "6.25",
                "available": "4.25",
                "debt": "0",
                "locked": "2",
                "bonus": "10",
                "interestBase": "0",
            },
            {
                "coin": "BGB",
                "equity": "1.5",
                "usdValue": "4.9",
                "balance": "1.25",
                "balanceOriginal": "1.25",
                "available": "1.25",
                "debt": "0",
                "locked": "0",
                "bonus": "0",
                "interestBase": "0",
            },
        ],
    },
}

CLOSE_UNDER_ANOTHER_MODE = [
    pytest.param("isolated", MarginMode.CROSS, id="isolated-position-crossed-symbol"),
    pytest.param("crossed", MarginMode.ISOLATED, id="crossed-position-isolated-symbol"),
]

_BOT_FILE = """
secrets_file = "creds.toml"

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
fast_ma_length = 2
slow_ma_length = 3
total_balance_ratio = 0.1
tag = "p1"

[live.trading_account]
exchange = "{module}:plugged_exchange"
secret_name = "my_account"

[live.ohlcv_provider]
ohlcv_provider = "bitget"
"""

_CREDENTIALS = """
[[secrets]]
name = "my_account"
api_key = "shh"
"""

_plugged: dict[str, Any] = {}


def plugged_exchange(**settings: Any) -> Any:
    """The connector the test bot names in place of a venue."""
    return _plugged["exchange"]


def position_row(
    symbol: str,
    margin_mode: str,
    *,
    margin_coin: str = "USDT",
    pos_side: str = "long",
    liquidation_price: str = "77.0",
    margin: str = "12.5",
    unrealised_pnl: str = "0",
) -> dict:
    return {
        "category": f"{margin_coin}-FUTURES",
        "symbol": symbol,
        "marginCoin": margin_coin,
        "holdMode": "one_way_mode",
        "posSide": pos_side,
        "marginMode": margin_mode,
        "positionBalance": margin,
        "available": "1.5",
        "frozen": "0",
        "total": "1.5",
        "leverage": "14",
        "curRealisedPnl": "0",
        "avgPrice": "99.0",
        "positionStatus": "normal",
        "unrealisedPnl": unrealised_pnl,
        "liquidationPrice": liquidation_price,
        "mmr": "0.004",
        "markPrice": "100.0",
        "createdTime": "1700000000000",
        "updatedTime": "1700000000000",
    }


def positions_response(*rows: dict) -> dict:
    return {"code": "00000", "msg": "success", "data": {"list": list(rows) or None}}


def open_orders_response(rows: list[dict], cursor: str = "") -> dict:
    return {
        "code": "00000",
        "msg": "success",
        "data": {"list": rows or None, "cursor": cursor},
    }


def rejection(code: str, msg: str) -> Any:
    response = HttpResponse(
        status_code=400,
        reason_phrase="Bad Request",
        url="https://api.bitget.com/test",
        path="/test",
    )
    return BitgetHTTPError(response, code=code, msg=msg)


def posts_to(client: Mock, endpoint: str) -> list[dict]:
    return [
        call.args[1] for call in client.post.call_args_list if call.args[0] == endpoint
    ]


@pytest.fixture
def bitget_client() -> Mock:
    client = Mock(spec=BitgetClient)
    client.demo_trading = False
    return client


@pytest.fixture
def margin_account(bitget_client: Mock) -> dict[str, list[dict]]:
    """A venue holding the position rows a test adds.

    It refuses a reduce-only order naming another margin mode than its
    position's as `25227` and removes the position a matching one closes.
    """
    rows: dict[str, list[dict]] = {"positions": []}
    responses = {
        INSTRUMENTS_ENDPOINT: INSTRUMENTS_RESPONSE,
        SETTINGS_ENDPOINT: ACCOUNT_SETTINGS_RESPONSE,
        ASSETS_ENDPOINT: ASSETS_RESPONSE,
    }

    async def answer_get(endpoint, params=None):
        if endpoint == POSITIONS_ENDPOINT:
            return positions_response(
                *(
                    row
                    for row in rows["positions"]
                    if row["category"] == params["category"]
                )
            )
        if endpoint in (
            FILLS_ENDPOINT,
            HISTORY_ORDERS_ENDPOINT,
            UNFILLED_ORDERS_ENDPOINT,
        ):
            return open_orders_response([])
        return responses.get(endpoint, {"code": "00000", "msg": "success", "data": {}})

    async def answer_post(endpoint, params):
        if endpoint == PLACE_ORDER_ENDPOINT and params["reduceOnly"] == "yes":
            held = next(
                (
                    row
                    for row in rows["positions"]
                    if row["symbol"] == params["symbol"]
                    and row["marginMode"] == params["marginMode"]
                ),
                None,
            )
            if held is None:
                raise rejection(NO_POSITION_TO_CLOSE, "No position available to close")
            rows["positions"].remove(held)
        return {
            "code": "00000",
            "msg": "success",
            "data": {"orderId": "placed", "clientOid": params.get("clientOid", "")},
        }

    bitget_client.get.side_effect = answer_get
    bitget_client.post.side_effect = answer_post
    return rows


@pytest.fixture
async def unified_exchange(bitget_client: Mock, margin_account: dict) -> Any:
    return await BitgetExchange.create_from_client(bitget_client)


@pytest.fixture
def plugged(unified_exchange: Any) -> Iterator[Any]:
    _plugged["exchange"] = unified_exchange
    yield unified_exchange
    _plugged.clear()


@pytest.fixture
def bot_folder(tmp_path: Path, plugged: Any) -> Path:
    folder = tmp_path / "unified-bot-example"
    folder.mkdir()
    (folder / "creds.toml").write_text(_CREDENTIALS)
    (folder / "bot.toml").write_text(_BOT_FILE.format(module=__name__))
    return folder


@pytest.fixture
async def holding_a_position(
    plugged: Any,
    margin_account: dict[str, list[dict]],
    held: str,
    symbol_setting: MarginMode,
) -> None:
    margin_account["positions"].append(position_row("BTCUSDT", held))
    await plugged.set_margin_mode(BTC_USDT, symbol_setting)


@pytest.mark.parametrize(("held", "symbol_setting"), CLOSE_UNDER_ANOTHER_MODE)
def test_flatten_of_a_symbol_closes_it_under_the_positions_mode(
    bot_folder, holding_a_position, bitget_client, margin_account, held
):
    exit_code = flatten(bot_folder, "BTC/USDT:USDT", external=False)

    (body,) = posts_to(bitget_client, PLACE_ORDER_ENDPOINT)
    assert body["marginMode"] == held
    assert exit_code == 0
    assert margin_account["positions"] == []


@pytest.mark.parametrize(("held", "symbol_setting"), CLOSE_UNDER_ANOTHER_MODE)
def test_flatten_external_closes_what_no_profile_owns_under_the_positions_mode(
    bot_folder, holding_a_position, bitget_client, margin_account, held
):
    exit_code = flatten(bot_folder, None, external=True)

    (body,) = posts_to(bitget_client, PLACE_ORDER_ENDPOINT)
    assert body["marginMode"] == held
    assert body["reduceOnly"] == "yes"
    assert exit_code == 0
