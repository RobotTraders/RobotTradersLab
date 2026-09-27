import io
import sys

from robottraderslab.cli.console import use_utf8_console


def test_a_console_encoding_narrower_than_the_report(monkeypatch):
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", console)

    use_utf8_console()

    assert console.encoding == "utf-8"


def test_both_streams_are_widened(monkeypatch):
    errors = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stderr", errors)

    use_utf8_console()

    assert errors.encoding == "utf-8"


def test_a_stream_that_cannot_be_reconfigured(monkeypatch):
    captured = io.StringIO()
    errors = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", captured)
    monkeypatch.setattr("sys.stderr", errors)

    use_utf8_console()

    assert errors.encoding == "utf-8"
    assert sys.stdout is captured
