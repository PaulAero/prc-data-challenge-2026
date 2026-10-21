from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from prc.latent.regimes import SoftRegimeEncoder


UTC = timezone.utc


def test_regime_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(7)
    rows = 200
    state = pl.DataFrame(
        {
            "airport": ["LFPG"] * rows,
            "timestamp": [
                datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * index)
                for index in range(rows)
            ],
            "raw_latent_z_0": rng.normal(size=rows),
            "raw_latent_z_1": rng.normal(size=rows),
            "residual_latent_z_0": rng.normal(size=rows),
            "residual_latent_z_1": rng.normal(size=rows),
        }
    )
    encoder = SoftRegimeEncoder.fit(
        state,
        cutoff=datetime(2025, 1, 2, tzinfo=UTC),
        n_regimes=3,
    )
    transformed = encoder.transform(state)
    probabilities = transformed.select(pl.col("^regime_probability_.*$")).to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert transformed.get_column("regime").n_unique() <= 3
