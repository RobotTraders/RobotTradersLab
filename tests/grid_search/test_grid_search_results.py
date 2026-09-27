import pandas as pd
import pytest

from robottraderslab.grid_search import GridSearchResults
from robottraderslab.grid_search.grid_point import GridPoint
from robottraderslab.grid_search.optimisation_result import OptimisationResult

FAILURE = "ProfileConfig.__init__() got an unexpected keyword argument '0'"


def _point(fast: int, slow: int) -> GridPoint:
    return GridPoint(
        parameters={
            "strategy.profiles.fast_ma_length": fast,
            "strategy.profiles.slow_ma_length": slow,
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
            "strategy.profiles.fast_ma_length",
            "strategy.profiles.slow_ma_length",
        ]
        assert loaded.dataframe["error"].isna().tolist() == [True, False]

    def test_a_file_that_to_csv_did_not_write_names_itself(self, tmp_path):
        foreign = tmp_path / "from_somewhere_else.csv"
        foreign.write_text(
            "strategy.profiles.fast_ma_length,roi\n10,0.95\n", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="from_somewhere_else.csv"):
            GridSearchResults.from_csv(foreign)
