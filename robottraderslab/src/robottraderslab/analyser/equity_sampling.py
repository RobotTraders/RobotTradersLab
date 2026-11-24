import pandas as pd


def sample_equity_at(
    equity_curve: pd.Series, times: pd.Series, *, include_end: bool
) -> pd.Series:
    """The first point is always kept so a fall is measured from where the run started.

    The last point is kept where the caller draws the curve to its end.

    Args:
        equity_curve: Equity by timestamp.
        times: The moments to read, in any order and with repeats.
    """
    sampled = equity_curve.loc[equity_curve.index.isin(times.dropna())]
    anchors = [pd.Series({equity_curve.index[0]: equity_curve.iloc[0]}), sampled]
    if include_end:
        anchors.append(pd.Series({equity_curve.index[-1]: equity_curve.iloc[-1]}))
    combined = pd.concat(anchors)
    return combined[~combined.index.duplicated()].sort_index()
