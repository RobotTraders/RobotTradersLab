from dataclasses import dataclass

_FIRST_ERROR_STATUS = 400


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """The parts of a response an exchange plugin classifies errors from."""

    status_code: int
    reason_phrase: str
    url: str
    path: str

    @property
    def is_error(self) -> bool:
        return self.status_code >= _FIRST_ERROR_STATUS


@dataclass(frozen=True, slots=True)
class RequestToSign:
    """The parts of an outgoing request an exchange signature covers.

    Path and query carry their encoded forms, since an exchange recomputes a
    signature from the bytes it receives.
    """

    method: str
    path: str
    query: str
    body: bytes

    @property
    def path_with_query(self) -> str:
        return f"{self.path}?{self.query}" if self.query else self.path
