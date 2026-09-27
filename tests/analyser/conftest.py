from unittest.mock import Mock

import pytest

from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.chart import ChartService
from robottraderslab.plotting import PlottingService


@pytest.fixture
def ohlcv_provider() -> Mock:
    return Mock(spec=OHLCVProviderProtocol)


@pytest.fixture
def plotting_service() -> Mock:
    return Mock(spec=PlottingService)


@pytest.fixture
def chart_service() -> Mock:
    return Mock(spec=ChartService)
