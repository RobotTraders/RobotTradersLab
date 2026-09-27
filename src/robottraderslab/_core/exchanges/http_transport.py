import logging
import ssl
from collections.abc import Callable
from functools import cache
from typing import Any, cast

import httpx

from .http_messages import HttpResponse, RequestToSign

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
MAX_CONNECTIONS = 100
MAX_KEEPALIVE_CONNECTIONS = 50
KEEPALIVE_EXPIRY_SECONDS = 60.0

type JsonLike = dict[str, Any]
type ResponseClassifier = Callable[[HttpResponse, JsonLike], None]
type RequestSigner = Callable[[RequestToSign], dict[str, str]]


class HttpTransport:
    """Async HTTP transport shared by exchange clients.

    Owns the HTTP session, sends requests, and parses JSON bodies. Exchange
    plugins keep full control of error semantics through two hooks: a
    classifier that inspects every response and raises the exchange's own
    exceptions, and the exception type raised for connection-level failures.
    Authenticated exchanges add a third: a signer handed the request as it
    goes on the wire, returning the headers that authenticate it.
    """

    def __init__(
        self,
        session: httpx.AsyncClient,
        base_url: str,
        classify_response: ResponseClassifier,
        connection_error: type[Exception],
        sign_request: RequestSigner | None,
    ) -> None:
        """Use `create` unless a preconfigured session is needed.

        Args:
            session: Session used for all requests, configured with headers
                and timeout as the exchange requires.
            base_url: Base URL prepended to every request path.
            classify_response: Called with each response and its parsed JSON
                body; raises the exchange's exceptions for error responses.
            connection_error: Exception type raised on connection-level
                failures, constructed with the underlying error type and
                message.
            sign_request: Called with each request before it is sent; returns
                the authentication headers to add. Omitted for public
                endpoints, which need none.
        """
        self.session = session
        self._base_url = base_url.rstrip("/")
        self._classify_response = classify_response
        self._connection_error = connection_error
        self._sign_request = sign_request

    @classmethod
    def create(
        cls,
        base_url: str,
        *,
        classify_response: ResponseClassifier,
        connection_error: type[Exception],
        sign_request: RequestSigner | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> "HttpTransport":
        """Create a transport with its own session.

        Args:
            base_url: Base URL prepended to every request path.
            classify_response: Called with each response and its parsed JSON
                body; raises the exchange's exceptions for error responses.
            connection_error: Exception type raised on connection-level
                failures, constructed with the underlying error type and
                message.
            sign_request: Called with each request before it is sent; returns
                the authentication headers to add.
            headers: Optional headers sent with every request.
            timeout: Timeout in seconds for connect and read operations.
        """
        session = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            verify=_ssl_context(),
            limits=httpx.Limits(
                max_connections=MAX_CONNECTIONS,
                max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
                keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
            ),
        )
        return cls(session, base_url, classify_response, connection_error, sign_request)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        json: JsonLike | None = None,
    ) -> JsonLike:
        """Send a request and return the parsed JSON response body.

        Args:
            path: Endpoint path appended to the base URL.
            data: Optional form-encoded body.

        Returns:
            The parsed JSON body, or an empty dict if the body is not JSON.

        Raises:
            Exception: The configured connection error on connection-level
                failures, or whatever the response classifier raises.
        """
        request = self.session.build_request(
            method,
            self._base_url + path,
            params=params,
            data=data,
            json=json,
        )
        if self._sign_request is not None:
            request.headers.update(self._sign_request(_request_to_sign(request)))

        try:
            response = await self.session.send(request)
        except httpx.TransportError as e:
            message = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            raise self._connection_error(message) from e

        try:
            response_json = response.json()
        except ValueError:
            logger.warning(f"Failed to parse JSON response from {path}")
            response_json = {}

        self._classify_response(_http_response(response), response_json)
        return cast(JsonLike, response_json)


def _request_to_sign(request: httpx.Request) -> RequestToSign:
    path, _, query = request.url.raw_path.decode("ascii").partition("?")
    return RequestToSign(
        method=request.method,
        path=path,
        query=query,
        body=request.content,
    )


def _http_response(response: httpx.Response) -> HttpResponse:
    return HttpResponse(
        status_code=response.status_code,
        reason_phrase=response.reason_phrase,
        url=str(response.url),
        path=response.url.path,
    )


@cache
def _ssl_context() -> ssl.SSLContext:
    """Loading the certificate bundle is the whole cost of building a client."""
    return httpx.create_ssl_context()
