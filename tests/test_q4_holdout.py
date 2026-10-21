import numpy as np
import polars as pl

from prc.timeseries.q4_holdout import weighted_bucket_rmse


def test_weighted_bucket_rmse_ignores_empty_and_missing_bins() -> None:
    actual = np.array([10.0, 20.0, np.nan, 50.0])
    predicted = np.array([8.0, 24.0, 30.0, 999.0])
    counts = np.array([1.0, 3.0, 2.0, 0.0])
    assert weighted_bucket_rmse(actual, predicted, counts) == np.sqrt(13.0)
