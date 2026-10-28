import pandas as pd
import pytest

from robottraderslab.grid_search import GridSearchResults
from robottraderslab.grid_search.grid_point import GridPoint
from robottraderslab.grid_search.optimisation_result import OptimisationResult

FAILURE = "ProfileConfig.__init__() got an unexpected keyword argument '0'"


def _point(fast: int, slow: int) -> GridPoint:
    return GridPoint(
        parameters={
            "fast_ma_length": fast,
            "slow_ma_length": slow,
        }
    )


def _scored(
    fast: int, slow: int, *, closed_trades: int, roi: float
) -> OptimisationResult:
    return OptimisationResult(
        grid_point=_point(fast, slow),
        roi=roi,
        closed_trades=closed_trades,
    )


class TestGridSearchResults:
    def test_a_failed_point_carries_what_it_raised(self):
        results = GridSearchResults.from_results(
            [OptimisationResult.from_error(_point(5, 20), FAILURE)]
        )

        assert results.dataframe["error"].tolist() == [FAILURE]

    def test_a_failed_point_carries_its_error_into_the_file(self, tmp_path):
        results = GridSearchResults.from_results(
            [OptimisationResult.from_error(_point(5, 20), FAILURE)]
        )
        exported = tmp_path / "results.csv"

        results.to_csv(exported)

        assert pd.read_csv(exported)["error"].tolist() == [FAILURE]

    def test_a_file_named_by_no_path_lands_under_the_bots_grid_searches(self, tmp_path):
        results = GridSearchResults.from_results(
            [_scored(5, 20, closed_trades=12, roi=0.31)], config_dir=tmp_path
        )

        exported = results.to_csv()

        assert exported.name == "results.csv"
        assert exported.parent.parent == tmp_path / "grid-searches"

    def test_a_point_that_scored_is_marked_by_no_error(self):
        results = GridSearchResults.from_results(
            [
                _scored(5, 20, closed_trades=12, roi=0.31),
                OptimisationResult.from_error(_point(10, 20), FAILURE),
                OptimisationResult.from_error(_point(15, 20), FAILURE),
            ]
        )

        assert results.dataframe["error"].isna().tolist() == [True, False, False]

    def test_a_point_whose_parameters_traded_nothing_scored(self):
        results = GridSearchResults.from_results(
            [_scored(5, 20, closed_trades=0, roi=0.0)]
        )

        assert results.dataframe["error"].isna().all()

    def test_an_exported_error_is_no_parameter_when_read_back(self, tmp_path):
        exported = tmp_path / "results.csv"
        GridSearchResults.from_results(
            [
                _scored(5, 20, closed_trades=12, roi=0.31),
                OptimisationResult.from_error(_point(10, 20), FAILURE),
            ]
        ).to_csv(exported)

        loaded = GridSearchResults.from_csv(exported)

        assert list(loaded.dataframe.columns)[:2] == [
            "fast_ma_length",
            "slow_ma_length",
        ]
        assert loaded.dataframe["error"].isna().tolist() == [True, False]

    @pytest.mark.parametrize(
        ("content", "named"),
        [
            ("fast_ma_length,roi\n10,0.95\n", "`error`"),
            (
                "fast_ma_length,slow_ma_length,"
                "roi,win_rate,max_drawdown,risk_reward_ratio,sharpe_ratio,"
                "closed_trades,error\n10,20,0.95,0.5,-0.2,1.5,1.8,12,\n",
                "`sortino_ratio`, `calmar_ratio`",
            ),
        ],
    )
    def test_a_file_lacking_a_column_to_csv_writes_names_it(
        self, tmp_path, content, named
    ):
        incomplete = tmp_path / "earlier_results.csv"
        incomplete.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match=named) as refusal:
            GridSearchResults.from_csv(incomplete)

        assert str(incomplete) in str(refusal.value)

    def test_a_file_holding_one_parameter_column_names_it(self, tmp_path):
        one_parameter = tmp_path / "one_parameter_results.csv"
        one_parameter.write_text(
            "fast_ma_length,roi,win_rate,max_drawdown,"
            "risk_reward_ratio,sharpe_ratio,sortino_ratio,calmar_ratio,"
            "closed_trades,error\n10,0.95,0.5,-0.2,1.5,1.8,2.4,3.1,12,\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="fast_ma_length") as refusal:
            GridSearchResults.from_csv(one_parameter)

        assert str(one_parameter) in str(refusal.value)
