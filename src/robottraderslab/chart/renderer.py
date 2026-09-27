import html
import logging
import re
import webbrowser
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from .encoding import encode_payload
from .payload import ChartPayload

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent / "assets"
_LIBRARY_ASSET = "lightweight-charts-5.2.0.standalone.production.js"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{[A-Z_]+\}\}")


def render_chart(payload: ChartPayload) -> str:
    """Render a payload into a self-contained HTML document.

    The charting library, styles and data are inlined, so the result renders
    with no network access and can be moved or shared as a single file.
    """
    substitutions = {
        "{{TITLE}}": html.escape(payload.title),
        "{{CSS}}": _read_asset("chart.css"),
        "{{LIBRARY}}": _read_asset(_LIBRARY_ASSET),
        "{{PAYLOAD}}": encode_payload(payload),
        "{{CHART_JS}}": _read_asset("chart.js"),
    }
    template = _read_asset("chart.html")
    return _PLACEHOLDER_PATTERN.sub(
        lambda match: substitutions[match.group()], template
    )


def write_chart(payload: ChartPayload, destination: Path) -> Path:
    """Write a rendered chart to disk and return where it landed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_chart(payload), encoding="utf-8")
    logger.info("Chart written to %s", destination)
    return destination


def write_charts(
    payloads: Sequence[ChartPayload], destinations: Sequence[Path]
) -> list[Path]:
    return [
        write_chart(payload, destination)
        for payload, destination in zip(payloads, destinations, strict=True)
    ]


def open_in_browser(destination: Path) -> None:
    """Open a chart that is already on disk in the default browser."""
    webbrowser.open(destination.as_uri())


@cache
def _read_asset(name: str) -> str:
    return (_ASSETS_DIR / name).read_text(encoding="utf-8")
