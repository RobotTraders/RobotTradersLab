from collections.abc import Mapping
from dataclasses import dataclass, field

_FIRST_ERROR_STATUS = 400


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """The parts of a response an exchange plugin reads, to classify errors and
    to take in what the venue announces.

    Attributes:
        headers: Looked up without regard to case when the transport built
            the response.
    """

    status_code: int
    reason_phrase: str
    url: str
    path: str
    headers: Mapping[str, str] = field(default_factory=dict)

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
