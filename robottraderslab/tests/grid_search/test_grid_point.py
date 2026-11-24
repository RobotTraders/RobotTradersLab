from robottraderslab.grid_search.grid_point import GridPoint


class TestGridPoint:
    """Test GridPoint model."""

    def test_stores_parameters(self):
        point = GridPoint(parameters={"param1": 5.0, "param2": 10.0})

        assert point.parameters == {"param1": 5.0, "param2": 10.0}

    def test_repr(self):
        point = GridPoint(parameters={"trix_length": 8.0, "signal_length": 15.0})

        repr_str = repr(point)

        assert "GridPoint" in repr_str
        assert "trix_length=8.0" in repr_str
        assert "signal_length=15.0" in repr_str

    def test_with_int_values(self):
        point = GridPoint(parameters={"fast_ma": 3, "slow_ma": 7})

        assert point.parameters["fast_ma"] == 3
        assert isinstance(point.parameters["fast_ma"], int)

    def test_with_mixed_types(self):
        point = GridPoint(parameters={"length": 5, "ratio": 0.5})

        assert isinstance(point.parameters["length"], int)
        assert isinstance(point.parameters["ratio"], float)
