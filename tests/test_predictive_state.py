from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from prc.latent.predictive_state import PredictiveLatentEncoder


UTC = timezone.utc


def _state() -> pl.DataFrame:
    timestamps = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i) for i in range(80)]
    return pl.DataFrame(
        {
            "airport": ["LFPG"] * 80,
            "timestamp": timestamps,
            "departure_count": [i % 7 for i in range(80)],
            "arrival_count": [(i + 2) % 5 for i in range(80)],
            "scheduled_departure_count": [(i + 1) % 8 for i in range(80)],
            "taxi_out_mean_sec": [600.0 + 3 * (i % 9) for i in range(80)],
        }
    )


def test_holdout_targets_do_not_change_fitted_latent_state() -> None:
    state = _state()
    cutoff = datetime(2025, 1, 1, 12, tzinfo=UTC)
    changed = state.with_columns(
        pl.when(pl.col("timestamp") >= cutoff)
        .then(pl.lit(999_999.0))
        .otherwise(pl.col("taxi_out_mean_sec"))
        .alias("taxi_out_mean_sec")
    )
    first = PredictiveLatentEncoder.fit(
        state, cutoff=cutoff, n_components=2, horizons=(1, 2)
    ).transform(state)
    second = PredictiveLatentEncoder.fit(
        changed, cutoff=cutoff, n_components=2, horizons=(1, 2)
    ).transform(changed)
    np.testing.assert_allclose(
        first.select("latent_z_0", "latent_z_1").to_numpy(),
        second.select("latent_z_0", "latent_z_1").to_numpy(),
        atol=1e-5,
    )
