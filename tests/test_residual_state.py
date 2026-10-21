from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from prc.latent.residual_state import SharedResidualPredictiveEncoder


UTC = timezone.utc


def _state() -> pl.DataFrame:
    rows = []
    for airport, offset in (("LFPG", 0), ("EDDF", 5)):
        for index in range(100):
            rows.append(
                {
                    "airport": airport,
                    "timestamp": datetime(2025, 1, 1, tzinfo=UTC)
                    + timedelta(minutes=15 * index),
                    "departure_count": index % 7 + offset,
                    "arrival_count": (index + 2) % 5 + offset,
                    "scheduled_departure_count": (index + 1) % 8 + offset,
                    "taxi_out_mean_sec": 600.0 + 3 * (index % 9) + 20 * offset,
                }
            )
    return pl.DataFrame(rows)


def test_residual_encoder_never_reads_holdout_targets() -> None:
    state = _state()
    cutoff = datetime(2025, 1, 1, 18, tzinfo=UTC)
    changed = state.with_columns(
        pl.when(pl.col("timestamp") >= cutoff)
        .then(pl.lit(999_999.0))
        .otherwise(pl.col("taxi_out_mean_sec"))
        .alias("taxi_out_mean_sec")
    )
    first = SharedResidualPredictiveEncoder.fit(
        state, cutoff=cutoff, n_components=2, horizons=(1, 2)
    ).transform(state)
    second = SharedResidualPredictiveEncoder.fit(
        changed, cutoff=cutoff, n_components=2, horizons=(1, 2)
    ).transform(changed)
    np.testing.assert_allclose(
        first.select("latent_z_0", "latent_z_1").to_numpy(),
        second.select("latent_z_0", "latent_z_1").to_numpy(),
        atol=1e-5,
    )
