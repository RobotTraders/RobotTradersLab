from unittest.mock import Mock

import pytest

from robottraderslab.chart import ChartService
from robottraderslab.plotting import PlottingService
from robottraderslab_core.exchanges import OHLCVProviderProtocol


@pytest.fixture
def ohlcv_provider() -> Mock:
    return Mock(spec=OHLCVProviderProtocol)


@pytest.fixture
def plotting_service() -> Mock:
    return Mock(spec=PlottingService)


@pytest.fixture
def chart_service() -> Mock:
    return Mock(spec=ChartService)
