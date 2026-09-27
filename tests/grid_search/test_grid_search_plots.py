import pandas as pd
import pytest

from robottraderslab.grid_search.plots import plot_heatmap, plot_heatmap_summary

_TRIX_LENGTH = "strategy.profiles.trix_length"
_SIGNAL_LENGTH = "strategy.profiles.signal_length"


@pytest.fixture
def grid_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            _TRIX_LENGTH: [5, 5, 5, 10, 10, 10, 15, 15, 15],
            _SIGNAL_LENGTH: [10, 20, 30, 10, 20, 30, 10, 20, 30],
            "sharpe_ratio": [0.5, 1.2, 0.8, 1.5, 2.0, 1.3, 0.9, 1.1, 0.7],
            "roi": [5.0, 12.0, 8.0, 15.0, 20.0, 13.0, 9.0, 11.0, 7.0],
            "max_drawdown": [-10.0, -5.0, -8.0, -3.0, -2.0, -4.0, -7.0, -6.0, -9.0],
            "win_rate": [45.0, 55.0, 50.0, 60.0, 65.0, 58.0, 48.0, 52.0, 47.0],
            "risk_reward_ratio": [0.8, 1.5, 1.0, 1.8, 2.5, 1.6, 1.0, 1.3, 0.9],
            "closed_trades": [30, 25, 28, 35, 40, 33, 22, 27, 31],
        }
    )


class TestPlotHeatmap:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_the_default_title_is_the_metric_display_name(self, grid_results):
        figure = plot_heatmap(
            grid_results, _TRIX_LENGTH, _SIGNAL_LENGTH, "sharpe_ratio"
        )

        assert figure.axes[0].get_title() == "Sharpe Ratio"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_the_cells_carry_no_annotations(self, grid_results):
        figure = plot_heatmap(
            grid_results, _TRIX_LENGTH, _SIGNAL_LENGTH, "sharpe_ratio"
        )

        assert list(figure.axes[0].texts) == []

    def test_the_metric_is_saved_under_its_own_name(self, grid_results, tmp_path):
        plot_heatmap(
            grid_results,
            _TRIX_LENGTH,
            _SIGNAL_LENGTH,
            "sharpe_ratio",
            save_to_path=tmp_path,
        )

        assert (tmp_path / "sharpe_ratio.png").exists()

    def test_a_metric_with_gaps_is_smoothed_titled_and_saved(
        self, grid_results, tmp_path
    ):
        grid_results.loc[0, "sharpe_ratio"] = None

        figure = plot_heatmap(
            grid_results,
            _TRIX_LENGTH,
            _SIGNAL_LENGTH,
            "sharpe_ratio",
            smoothing=1.0,
            title="Custom Title",
            filename="heatmap_test",
            save_to_path=tmp_path,
        )

        assert (tmp_path / "heatmap_test.png").exists()
        assert figure.axes[0].get_title() == "Custom Title"


class TestPlotHeatmapSummary:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_every_summary_metric_gets_a_panel(self, grid_results):
        figure = plot_heatmap_summary(grid_results, _TRIX_LENGTH, _SIGNAL_LENGTH)

        titles = [axes.get_title() for axes in figure.axes if axes.get_title()]
        assert titles == [
            "Roi",
            "Win Rate",
            "Max Drawdown",
            "Risk Reward Ratio",
            "Sharpe Ratio",
            "Closed Trades",
        ]

    def test_the_six_metrics_are_saved_under_the_default_name(
        self, grid_results, tmp_path
    ):
        plot_heatmap_summary(
            grid_results,
            _TRIX_LENGTH,
            _SIGNAL_LENGTH,
            smoothing=1.0,
            title="Sweep",
            save_to_path=tmp_path,
        )

        assert (tmp_path / "heatmap_summary.png").exists()
