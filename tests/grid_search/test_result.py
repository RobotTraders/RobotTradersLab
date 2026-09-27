from unittest.mock import Mock

from robottraderslab.analyser import Analyser, SummaryMetrics
from robottraderslab.backtester import BacktestOutputs
from robottraderslab.grid_search.grid_point import GridPoint
from robottraderslab.grid_search.optimisation_result import OptimisationResult


class TestOptimisationResult:
    def test_successful_result(self):
        point = GridPoint(parameters={"param1": 5.0, "param2": 10.0})

        successful = OptimisationResult(
            grid_point=point,
            error=None,
        )

        assert successful.is_successful
        assert successful.parameters == {"param1": 5.0, "param2": 10.0}

    def test_failed_result(self):
        point = GridPoint(parameters={"param1": 5.0, "param2": 10.0})

        failed = OptimisationResult(
            grid_point=point,
            error="Division by zero in calculation",
        )

        assert not failed.is_successful
        assert failed.error == "Division by zero in calculation"


class TestFromError:
    def test_from_error(self):
        point = GridPoint(parameters={"length": 10.0})

        error_result = OptimisationResult.from_error(point, "Strategy crashed")

        assert not error_result.is_successful
        assert error_result.error == "Strategy crashed"
        assert error_result.grid_point is point
        assert error_result.sharpe_ratio is None
        assert error_result.roi is None


class TestFromBacktest:
    def test_from_backtest_extracts_metrics_from_analyser(self):
        point = GridPoint(parameters={"length": 10.0})
        summary = SummaryMetrics(
            sharpe_ratio=1.85,
            roi=0.092,
            max_drawdown=-0.037,
            win_rate=0.667,
            risk_reward_ratio=2.15,
            closed_trades=42,
        )
        mock_analyser = Mock(spec=Analyser)
        mock_analyser.get_summary_metrics.return_value = summary
        mock_outputs = Mock(spec=BacktestOutputs)
        mock_outputs.create_analyser.return_value = mock_analyser

        optimisation_result = OptimisationResult.from_backtest(point, mock_outputs)

        assert optimisation_result.is_successful
        assert optimisation_result.grid_point is point
        assert optimisation_result.sharpe_ratio == 1.85
        assert optimisation_result.roi == 0.092
        assert optimisation_result.max_drawdown == -0.037
        assert optimisation_result.win_rate == 0.667
        assert optimisation_result.risk_reward_ratio == 2.15
        assert optimisation_result.closed_trades == 42

    def test_a_grid_point_saves_no_run_folder(self):
        point = GridPoint(parameters={"length": 10.0})
        mock_outputs = Mock(spec=BacktestOutputs)
        mock_outputs.create_analyser.return_value = Mock(spec=Analyser)

        OptimisationResult.from_backtest(point, mock_outputs)

        mock_outputs.create_analyser.assert_called_once_with(save=False)
