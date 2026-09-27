import pytest

from robottraderslab.exchanges import HttpResponse, RequestToSign


def _response(status_code: int) -> HttpResponse:
    return HttpResponse(
        status_code=status_code,
        reason_phrase="Bad Request",
        url="https://api.example.com/v1/data?limit=10",
        path="/v1/data",
    )


class TestHttpResponse:
    @pytest.mark.parametrize("status_code", [200, 201, 302, 399])
    def test_below_the_client_error_range_is_not_an_error(self, status_code):
        assert not _response(status_code).is_error

    @pytest.mark.parametrize("status_code", [400, 429, 500, 599])
    def test_client_and_server_ranges_are_errors(self, status_code):
        assert _response(status_code).is_error

    def test_query_is_appended_to_the_signed_path(self):
        request = RequestToSign(
            method="GET",
            path="/api/v2/mix/market/candles",
            query="symbol=BTCUSDT&limit=1000",
            body=b"",
        )

        assert (
            request.path_with_query
            == "/api/v2/mix/market/candles?symbol=BTCUSDT&limit=1000"
        )

    def test_signed_path_is_bare_without_a_query(self):
        request = RequestToSign(
            method="POST", path="/api/v2/mix/order/place-order", query="", body=b"{}"
        )

        assert request.path_with_query == "/api/v2/mix/order/place-order"
