from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from robottraderslab.exchanges import (
    HttpResponse,
    HttpTransport,
    JsonLike,
    RequestToSign,
)

BASE_URL = "https://api.example.com"


class FakeConnectionError(Exception):
    pass


class FakeResponseError(Exception):
    pass


class RecordingSigner:
    def __init__(self) -> None:
        self.signed: list[RequestToSign] = []

    def __call__(self, request: RequestToSign) -> dict[str, str]:
        self.signed.append(request)
        return {"x-signature": request.path_with_query}


def _classify(response: HttpResponse, response_json: JsonLike) -> None:
    if response.is_error:
        raise FakeResponseError(response_json.get("error", "unknown"))


def _make_response(
    status_code: int = 200, body: str = '{"data": "ok"}'
) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=body.encode(),
        request=httpx.Request("GET", f"{BASE_URL}/v1/data"),
    )


@pytest.fixture
def session() -> Mock:
    session = Mock(spec=httpx.AsyncClient)
    session.build_request = httpx.Request
    session.send = AsyncMock(return_value=_make_response())
    return session


@pytest.fixture
def transport(session) -> HttpTransport:
    return HttpTransport(
        session,
        BASE_URL,
        classify_response=_classify,
        connection_error=FakeConnectionError,
        sign_request=None,
    )


@pytest.fixture
def signer() -> RecordingSigner:
    return RecordingSigner()


@pytest.fixture
def signing_transport(session, signer) -> HttpTransport:
    return HttpTransport(
        session,
        BASE_URL,
        classify_response=_classify,
        connection_error=FakeConnectionError,
        sign_request=signer,
    )


async def test_returns_parsed_json(transport):
    response_json = await transport.request("GET", "/v1/data")

    assert response_json == {"data": "ok"}


async def test_prepends_base_url_to_path(transport, session):
    await transport.request("GET", "/v1/data", params={"limit": "10"})

    sent_request = session.send.call_args.args[0]
    assert str(sent_request.url) == f"{BASE_URL}/v1/data?limit=10"


async def test_connection_failure(transport, session):
    session.send.side_effect = httpx.ConnectError("connection refused")

    with pytest.raises(FakeConnectionError, match="ConnectError: connection refused"):
        await transport.request("GET", "/v1/data")


async def test_connection_failure_without_message_names_error_type(transport, session):
    session.send.side_effect = httpx.ReadTimeout("")

    with pytest.raises(FakeConnectionError, match="^ReadTimeout$"):
        await transport.request("GET", "/v1/data")


async def test_unparseable_body_returns_empty_dict(transport, session):
    session.send.return_value = _make_response(body="")

    response_json = await transport.request("GET", "/v1/data")

    assert response_json == {}


async def test_error_response_raises_via_classifier(transport, session):
    session.send.return_value = _make_response(
        status_code=400, body='{"error": "bad request"}'
    )

    with pytest.raises(FakeResponseError, match="bad request"):
        await transport.request("GET", "/v1/data")


async def test_classifier_reads_the_status_and_url_of_the_response(session):
    classified: list[HttpResponse] = []
    transport = HttpTransport(
        session,
        BASE_URL,
        classify_response=lambda response, _: classified.append(response),
        connection_error=FakeConnectionError,
        sign_request=None,
    )

    await transport.request("GET", "/v1/data")

    assert classified == [
        HttpResponse(
            status_code=200,
            reason_phrase="OK",
            url=f"{BASE_URL}/v1/data",
            path="/v1/data",
        )
    ]


async def test_signature_headers_are_sent(signing_transport, session):
    await signing_transport.request("GET", "/v1/data", params={"limit": "10"})

    sent_request = session.send.call_args.args[0]
    assert sent_request.headers["x-signature"] == "/v1/data?limit=10"


async def test_signer_reads_the_request_as_it_goes_on_the_wire(
    signing_transport, signer
):
    await signing_transport.request(
        "POST", "/v1/orders", params={"symbol": "BTC"}, json={"size": "1"}
    )

    assert signer.signed == [
        RequestToSign(
            method="POST", path="/v1/orders", query="symbol=BTC", body=b'{"size":"1"}'
        )
    ]
