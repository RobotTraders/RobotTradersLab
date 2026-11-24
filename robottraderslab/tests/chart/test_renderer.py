import logging
from dataclasses import replace

from robottraderslab.chart.renderer import render_chart, write_chart, write_charts


class TestRenderChart:
    def test_every_template_placeholder_is_substituted(self, payload):
        document = render_chart(payload)

        assert "{{" not in document

    def test_charting_library_is_inlined(self, payload):
        document = render_chart(payload)

        assert "TradingView Lightweight Charts" in document
        assert "window.LightweightCharts" in document

    def test_document_references_no_external_resource(self, payload):
        document = render_chart(payload)

        assert 'src="http' not in document
        assert 'href="http' not in document

    def test_payload_is_recoverable_from_the_document(self, payload, embedded_payload):
        document = render_chart(payload)

        assert embedded_payload(document)["markets"][0]["label"] == "BTC/USDT:USDT · 1h"

    def test_title_reaches_the_document_head(self, payload):
        document = render_chart(payload)

        assert "<title>Impulse backtest</title>" in document

    def test_markup_in_the_title_cannot_close_the_head(self, payload):
        titled = replace(payload, title="</title><script>alert(1)</script>")

        document = render_chart(titled)

        assert "<script>alert(1)</script>" not in document


class TestWriteChart:
    def test_creates_missing_parent_directories(self, payload, tmp_path):
        destination = tmp_path / "reports" / "nested" / "report.html"

        written = write_chart(payload, destination)

        assert written.exists()

    def test_written_document_matches_the_rendered_one(self, payload, tmp_path):
        destination = tmp_path / "report.html"

        written = write_chart(payload, destination)

        assert written.read_text(encoding="utf-8") == render_chart(payload)

    def test_overwrites_an_existing_report(self, payload, tmp_path):
        destination = tmp_path / "report.html"
        destination.write_text("stale", encoding="utf-8")

        written = write_chart(payload, destination)

        assert "stale" not in written.read_text(encoding="utf-8")

    def test_written_path_is_logged(self, payload, tmp_path, caplog):
        destination = tmp_path / "report.html"

        with caplog.at_level(logging.INFO):
            write_chart(payload, destination)

        assert str(destination) in caplog.text


class TestWriteCharts:
    def test_every_document_lands_on_disk(self, payload, tmp_path):
        destinations = [tmp_path / f"report_{index}.html" for index in range(5)]

        write_charts([payload] * len(destinations), destinations)

        assert all(destination.exists() for destination in destinations)

    def test_paths_come_back_in_the_order_requested(self, payload, tmp_path):
        destinations = [tmp_path / f"report_{index}.html" for index in range(5)]

        written = write_charts([payload] * len(destinations), destinations)

        assert written == destinations

    def test_each_document_holds_its_own_payload(self, payload, tmp_path):
        payloads = [
            replace(payload, title=title) for title in ("First chart", "Second chart")
        ]
        destinations = [tmp_path / "first.html", tmp_path / "second.html"]

        written = write_charts(payloads, destinations)

        assert "<title>Second chart</title>" in written[1].read_text(encoding="utf-8")

    def test_nothing_to_write(self, tmp_path):
        written = write_charts([], [])

        assert written == []
