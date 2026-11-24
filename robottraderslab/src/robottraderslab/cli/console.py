import io
import sys


def use_utf8_console() -> None:
    """Print every character a report can hold, whatever code page the console uses.

    A console that encodes in the machine's legacy code page raises on the
    first character outside it, after the work that produced the line is done.
    A line written to standard output reaches a pipe or a file before anything
    written to standard error after it.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", line_buffering=True)
