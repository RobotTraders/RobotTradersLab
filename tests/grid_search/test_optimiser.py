import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any
from unittest.mock import Mock, patch

import pytest

from robottraderslab import BacktestOutputs
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.grid_search.grid_point import GridPoint
from robottraderslab.grid_search.optimisation_result import OptimisationResult
from robottraderslab.grid_search.optimiser import (
    GridSearchOptimiser,
    _apply_grid_point_to_config,
    _measure_point,
    _read_nested_value,
)


class TestGridSearchOptimiser:
    @pytest.fixture
    def mock_bot_config(self) -> BotConfig:
        return BotConfig.model_validate(
            {
                "strategy": {
                    "strategy_class": "impulse",
                    "profiles": [
                        {
                            "symbol": "BTC/USDT:USDT",
                            "timeframe": "5m",
                            "total_balance_ratio": 1.0,
                            "param1": 0.0,
                            "param2": 0.0,
                        }
                    ],
                },
                "backtest": {
                    "initial_balance": {"USDT": 1000.0},
                    "maker_fee_rate": 0.0,
                    "taker_fee_rate": 0.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                    "ohlcv_provider": {},
                },
                "optimisation": {
                    "parameter_configs": [
                        {
                            "parameter": "strategy.profiles.param1",
                            "sampling": "linear",
                            "min": 5.0,
                            "max": 10.0,
                            "num_samples": 2,
                        },
                        {
                            "parameter": "strategy.profiles.param2",
                            "sampling": "linear",
                            "min": 10.0,
                            "max": 20.0,
                            "num_samples": 2,
                        },
                    ],
                    "max_workers": 1,
                },
            }
        )

    @pytest.fixture
    def mock_backtest_outputs(self):
        import pandas as pd

        outputs = Mock()
        outputs.final_balance = {"USDT": 1100.0}
        outputs.fills = pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "fill_type",
                "quantity",
                "price",
                "fee",
                "balance_before",
                "balance_after",
                "net_quantity",
                "side",
                "gross_quantity",
            ]
        )
        outputs.ohlcvs_data = {}
        return outputs

    def test_initializes_with_bot_config(self, mock_bot_config):
        optimiser = GridSearchOptimiser(mock_bot_config)

        assert optimiser._bot_config == mock_bot_config
        assert optimiser._optimisation_config is not None

    def test_run_evaluates_all_grid_points(
        self, mock_bot_config, mock_backtest_outputs
    ):
        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest"
        ) as mock_run_backtest:
            mock_run_backtest.return_value = mock_backtest_outputs

            with patch(
                "robottraderslab.grid_search.optimisation_result.OptimisationResult.from_backtest"
            ) as mock_from_backtest:

                def create_result(grid_point, backtest_outputs):
                    return OptimisationResult(
                        grid_point=grid_point,
                        error=None,
                        sharpe_ratio=1.5,
                        roi=0.15,
                        max_drawdown=0.05,
                        win_rate=0.60,
                        risk_reward_ratio=2.0,
                    )

                mock_from_backtest.side_effect = create_result

                optimiser = GridSearchOptimiser(mock_bot_config)
                results = optimiser.run()

                assert len(results) == 4
                assert mock_run_backtest.call_count == 4

                for result in results:
                    assert result.is_successful
                    assert result.error is None

    def test_run_handles_backtest_errors(self, mock_bot_config):
        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest"
        ) as mock_run_backtest:
            mock_run_backtest.side_effect = Exception("Backtest failed")

            optimiser = GridSearchOptimiser(mock_bot_config)
            results = optimiser.run()

            assert len(results) == 4

            for result in results:
                assert not result.is_successful
                assert result.error == "Backtest failed"

    def test_a_critical_stops_the_sequential_sweep(self, mock_bot_config):
        with (
            patch(
                "robottraderslab.grid_search.optimiser.run_backtest",
                side_effect=StrategyCriticalError("no candles declared"),
            ),
            pytest.raises(StrategyCriticalError, match="no candles declared"),
        ):
            GridSearchOptimiser(mock_bot_config).run()

    def test_run_handles_partial_failures(self, mock_bot_config, mock_backtest_outputs):
        call_count = 0

        def backtest_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Failed")
            return mock_backtest_outputs

        def create_result(grid_point, backtest_outputs):
            return OptimisationResult(
                grid_point=grid_point,
                error=None,
                sharpe_ratio=1.5,
                roi=0.15,
                max_drawdown=0.05,
                win_rate=0.60,
                risk_reward_ratio=2.0,
            )

        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest"
        ) as mock_run_backtest:
            mock_run_backtest.side_effect = backtest_side_effect

            with patch(
                "robottraderslab.grid_search.optimisation_result.OptimisationResult.from_backtest"
            ) as mock_from_backtest:
                mock_from_backtest.side_effect = create_result

                optimiser = GridSearchOptimiser(mock_bot_config)
                results = optimiser.run()

                assert len(results) == 4

                successful = [r for r in results if r.is_successful]
                failed = [r for r in results if not r.is_successful]

                assert len(successful) == 2
                assert len(failed) == 2

    def test_apply_grid_point_merges_parameters(self):
        config_dict = {
            "backtest": {
                "initial_balance": {"USDT": 1000.0},
                "maker_fee_rate": 0.0,
                "taker_fee_rate": 0.0,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "ohlcv_provider": {},
            },
            "strategy": {
                "strategy_class": "impulse",
                "profiles": [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "timeframe": "5m",
                        "total_balance_ratio": 1.0,
                        "param1": 0.0,
                        "param2": 0.0,
                    }
                ],
            },
        }
        grid_point = GridPoint(
            parameters={
                "strategy.profiles.param1": 5,
                "strategy.profiles.param2": 10,
            }
        )

        bot_config = _apply_grid_point_to_config(config_dict, grid_point)

        profile = bot_config.strategy.profiles[0]
        assert profile["param1"] == 5
        assert profile["param2"] == 10

    def test_apply_grid_point_updates_all_profiles(self):
        config_dict = {
            "backtest": {
                "initial_balance": {"USDT": 1000.0},
                "maker_fee_rate": 0.0,
                "taker_fee_rate": 0.0,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "ohlcv_provider": {},
            },
            "strategy": {
                "strategy_class": "impulse",
                "profiles": [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "timeframe": "5m",
                        "param1": 0.0,
                        "param2": 0.0,
                    },
                    {
                        "symbol": "ETH/USDT:USDT",
                        "timeframe": "15m",
                        "param1": 0.0,
                        "param2": 0.0,
                    },
                ],
            },
        }
        grid_point = GridPoint(
            parameters={
                "strategy.profiles.param1": 7,
                "strategy.profiles.param2": 15,
            }
        )

        bot_config = _apply_grid_point_to_config(config_dict, grid_point)

        profiles = bot_config.strategy.profiles
        assert len(profiles) == 2
        for profile in profiles:
            assert profile["param1"] == 7
            assert profile["param2"] == 15

    def test_results_contain_grid_point_parameters(
        self, mock_bot_config, mock_backtest_outputs
    ):
        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest"
        ) as mock_run_backtest:
            mock_run_backtest.return_value = mock_backtest_outputs

            optimiser = GridSearchOptimiser(mock_bot_config)
            results = optimiser.run()

            expected_param1_values = [5.0, 10.0]
            expected_param2_values = [10.0, 20.0]

            for result in results:
                assert (
                    result.parameters["strategy.profiles.param1"]
                    in expected_param1_values
                )
                assert (
                    result.parameters["strategy.profiles.param2"]
                    in expected_param2_values
                )

            param_combinations = set()
            for result in results:
                param_combinations.add(
                    (
                        result.parameters["strategy.profiles.param1"],
                        result.parameters["strategy.profiles.param2"],
                    )
                )

            assert len(param_combinations) == 4


class TestDottedPathFeature:
    @pytest.fixture
    def mock_result(self) -> OptimisationResult:
        return OptimisationResult(
            grid_point=GridPoint(parameters={}),
            error=None,
            sharpe_ratio=1.0,
            roi=0.1,
            max_drawdown=0.05,
            win_rate=0.6,
            risk_reward_ratio=2.0,
        )

    def test_sets_values_in_all_list_items(self, mock_result):
        bot_config = BotConfig.model_validate(
            {
                "strategy": {
                    "strategy_class": "impulse",
                    "profiles": [
                        {"symbol": "BTC/USDT:USDT", "param1": 0, "param2": 0},
                        {"symbol": "ETH/USDT:USDT", "param1": 0, "param2": 0},
                        {"symbol": "SOL/USDT:USDT", "param1": 0, "param2": 0},
                    ],
                },
                "backtest": {
                    "initial_balance": {"USDT": 1000.0},
                    "maker_fee_rate": 0.0,
                    "taker_fee_rate": 0.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                    "ohlcv_provider": {},
                },
                "optimisation": {
                    "parameter_configs": [
                        {
                            "parameter": "strategy.profiles.param1",
                            "sampling": "explicit",
                            "values": [10.0, 20.0],
                        },
                        {
                            "parameter": "strategy.profiles.param2",
                            "sampling": "explicit",
                            "values": [30.0, 40.0],
                        },
                    ],
                    "max_workers": 1,
                },
            }
        )

        captured_config = []

        def capture_and_return_result(bot_config):
            captured_config.append(bot_config.model_dump())
            return Mock()

        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest"
        ) as mock_backtest:
            mock_backtest.side_effect = capture_and_return_result

            with patch(
                "robottraderslab.grid_search.optimisation_result.OptimisationResult.from_backtest"
            ) as mock_from_backtest:
                mock_from_backtest.return_value = mock_result

                optimiser = GridSearchOptimiser(bot_config)
                optimiser.run()

                first_call = captured_config[0]
                profiles = first_call["strategy"]["profiles"]

                assert len(profiles) == 3
                for profile in profiles:
                    assert profile["param1"] == 10
                    assert profile["param2"] == 30

    @pytest.fixture
    def envelope_config_dict(self) -> dict[str, Any]:
        return {
            "backtest": {
                "initial_balance": {"USDT": 1000.0},
                "maker_fee_rate": 0.0,
                "taker_fee_rate": 0.0,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "ohlcv_provider": {},
            },
            "strategy": {
                "strategy_class": "envelope",
                "profiles": [
                    {
                        "symbol": "HYPE/USDT:USDT",
                        "timeframe": "15m",
                        "envelopes": [0.02, 0.03],
                    },
                    {
                        "symbol": "FET/USDT:USDT",
                        "timeframe": "15m",
                        "envelopes": [0.02, 0.03],
                    },
                ],
            },
        }

    def test_sweeps_both_rungs_of_a_list_valued_setting(self, envelope_config_dict):
        bot_config = BotConfig.model_validate(
            {
                **envelope_config_dict,
                "optimisation": {
                    "parameter_configs": [
                        {
                            "parameter": "strategy.profiles.envelopes.0",
                            "sampling": "explicit",
                            "values": [0.01, 0.015],
                        },
                        {
                            "parameter": "strategy.profiles.envelopes.1",
                            "sampling": "explicit",
                            "values": [0.025, 0.04],
                        },
                    ],
                    "max_workers": 1,
                },
            }
        )

        swept_envelopes = []

        def capture_and_return_outputs(patched_config):
            swept_envelopes.append(
                [profile["envelopes"] for profile in patched_config.strategy.profiles]
            )
            return Mock(spec=BacktestOutputs)

        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest",
            side_effect=capture_and_return_outputs,
        ):
            GridSearchOptimiser(bot_config).run()

        assert swept_envelopes == [
            [[0.01, 0.025], [0.01, 0.025]],
            [[0.01, 0.04], [0.01, 0.04]],
            [[0.015, 0.025], [0.015, 0.025]],
            [[0.015, 0.04], [0.015, 0.04]],
        ]

    def test_sweeps_one_list_element_named_by_its_index(self, envelope_config_dict):
        grid_point = GridPoint(parameters={"strategy.profiles.1.envelopes.0": 0.05})

        bot_config = _apply_grid_point_to_config(envelope_config_dict, grid_point)

        assert bot_config.strategy.profiles[0]["envelopes"] == [0.02, 0.03]
        assert bot_config.strategy.profiles[1]["envelopes"] == [0.05, 0.03]

    @pytest.mark.parametrize(
        "parameter_path",
        [
            "strategy.nonexistent",
            "strategy.profiles.nonexistent.4",
            "strategy.profiles.envelopes.5",
            "strategy.profiles.envelopes.0.7",
        ],
    )
    def test_path_matching_nothing_in_the_config(
        self, parameter_path, envelope_config_dict
    ):
        grid_point = GridPoint(parameters={parameter_path: 0.05})

        with pytest.raises(StrategyCriticalError, match=re.escape(parameter_path)):
            _apply_grid_point_to_config(envelope_config_dict, grid_point)

    def test_works_with_deeply_nested_structures(self, mock_result):
        bot_config = BotConfig.model_validate(
            {
                "strategy": {"strategy_class": "impulse", "profiles": []},
                "backtest": {
                    "initial_balance": {"USDT": 1000.0},
                    "maker_fee_rate": 0.0,
                    "taker_fee_rate": 0.0,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                    "ohlcv_provider": {"lookback_days": 0, "batch_size": 0},
                },
                "optimisation": {
                    "parameter_configs": [
                        {
                            "parameter": "backtest.ohlcv_provider.lookback_days",
                            "sampling": "explicit",
                            "values": [30.0, 60.0],
                        },
                        {
                            "parameter": "backtest.ohlcv_provider.batch_size",
                            "sampling": "explicit",
                            "values": [100.0, 200.0],
                        },
                    ],
                    "max_workers": 1,
                },
            }
        )

        captured_config = []

        def capture_and_return_result(bot_config):
            captured_config.append(bot_config.model_dump())
            return Mock()

        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest"
        ) as mock_backtest:
            mock_backtest.side_effect = capture_and_return_result

            with patch(
                "robottraderslab.grid_search.optimisation_result.OptimisationResult.from_backtest"
            ) as mock_from_backtest:
                mock_from_backtest.return_value = mock_result

                optimiser = GridSearchOptimiser(bot_config)
                optimiser.run()

                lookback_values = [
                    c["backtest"]["ohlcv_provider"]["lookback_days"]
                    for c in captured_config
                ]
                batch_values = [
                    c["backtest"]["ohlcv_provider"]["batch_size"]
                    for c in captured_config
                ]

                assert 30 in lookback_values
                assert 60 in lookback_values
                assert 100 in batch_values
                assert 200 in batch_values


class TestMeasurePoint:
    @pytest.fixture
    def grid_point(self) -> GridPoint:
        return GridPoint(
            parameters={
                "strategy.profiles.param1": 5.0,
                "strategy.profiles.param2": 10.0,
            }
        )

    @pytest.fixture
    def config_dict(self) -> dict:
        return {
            "backtest": {
                "initial_balance": {"USDT": 1000.0},
                "maker_fee_rate": 0.0,
                "taker_fee_rate": 0.0,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "ohlcv_provider": {},
            },
            "strategy": {
                "strategy_class": "impulse",
                "profiles": [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "timeframe": "5m",
                        "param1": 0.0,
                        "param2": 0.0,
                    }
                ],
            },
        }

    def test_preserves_all_metrics(self, config_dict, grid_point):
        backtest_result = OptimisationResult(
            grid_point=grid_point,
            error=None,
            sharpe_ratio=2.1,
            roi=0.35,
            max_drawdown=0.12,
            win_rate=0.55,
            risk_reward_ratio=1.8,
        )

        with (
            patch(
                "robottraderslab.grid_search.optimiser.run_backtest",
                return_value=Mock(),
            ),
            patch(
                "robottraderslab.grid_search.optimiser.OptimisationResult.from_backtest",
                return_value=backtest_result,
            ),
        ):
            worker_result = _measure_point(config_dict, grid_point)

        assert worker_result.sharpe_ratio == 2.1
        assert worker_result.roi == 0.35
        assert worker_result.max_drawdown == 0.12
        assert worker_result.win_rate == 0.55
        assert worker_result.risk_reward_ratio == 1.8
        assert worker_result.grid_point == grid_point

    def test_returns_error_result_on_backtest_failure(self, config_dict, grid_point):
        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest",
            side_effect=RuntimeError("boom"),
        ):
            worker_result = _measure_point(config_dict, grid_point)

        assert not worker_result.is_successful
        assert worker_result.error == "boom"

    def test_a_critical_stops_the_sweep(self, config_dict, grid_point):
        with (
            patch(
                "robottraderslab.grid_search.optimiser.run_backtest",
                side_effect=StrategyCriticalError("no candles declared"),
            ),
            pytest.raises(StrategyCriticalError, match="no candles declared"),
        ):
            _measure_point(config_dict, grid_point)

    def test_a_critical_crosses_the_process_boundary_with_its_class(self, config_dict):
        unknown_setting = GridPoint(
            parameters={"strategy.profiles.no_such_setting": 1.0}
        )

        with ProcessPoolExecutor(max_workers=1) as workers:
            measured = workers.submit(_measure_point, config_dict, unknown_setting)

            with pytest.raises(StrategyCriticalError, match="no_such_setting"):
                measured.result()

    def test_deep_copies_config_dict(self, config_dict, grid_point):
        original_profiles = config_dict["strategy"]["profiles"][0].copy()

        with patch(
            "robottraderslab.grid_search.optimiser.run_backtest",
            side_effect=RuntimeError("irrelevant"),
        ):
            _measure_point(config_dict, grid_point)

        assert config_dict["strategy"]["profiles"][0] == original_profiles


class TestReadingTheSweptValue:
    @pytest.fixture
    def config_dict(self) -> dict[str, Any]:
        return {
            "strategy": {
                "strategy_class": "impulse",
                "profiles": [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "trix_length": 8,
                        "bands": [0.02, 0.03],
                    },
                    {
                        "symbol": "ETH/USDT:USDT",
                        "trix_length": 12,
                        "bands": [0.04, 0.05],
                    },
                ],
            },
        }

    def test_a_path_fanning_out_over_a_list_reads_the_first_item(self, config_dict):
        assert _read_nested_value(config_dict, "strategy.profiles.trix_length") == 8

    def test_a_numeric_segment_reads_that_element(self, config_dict):
        assert _read_nested_value(config_dict, "strategy.profiles.1.bands.0") == 0.04

    def test_a_path_landing_on_a_table_is_refused(self, config_dict):
        with pytest.raises(StrategyCriticalError, match="holds no single value"):
            _read_nested_value(config_dict, "strategy.profiles")

    def test_a_path_fanning_out_over_an_empty_list_is_refused(self):
        with pytest.raises(StrategyCriticalError, match="empty list"):
            _read_nested_value(
                {"strategy": {"profiles": []}}, "strategy.profiles.trix_length"
            )

    def test_an_unset_setting_is_refused(self):
        with pytest.raises(StrategyCriticalError, match="leaves unset"):
            _read_nested_value({"strategy": {"tag": None}}, "strategy.tag")

    @pytest.mark.parametrize(
        ("path", "refusal"),
        [
            ("strategy.profiles.no_such_setting", "does not declare"),
            ("strategy.profiles.5.trix_length", "element 5 of a list holding 2"),
            ("strategy.profiles.trix_length.deeper", "under a single value"),
        ],
    )
    def test_a_path_matching_nothing_is_refused(self, config_dict, path, refusal):
        with pytest.raises(StrategyCriticalError, match=refusal):
            _read_nested_value(config_dict, path)
