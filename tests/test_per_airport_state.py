from datetime import datetime, timedelta, timezone

import polars as pl

from prc.latent.per_airport_state import PerAirportPredictiveLatentEncoder


UTC = timezone.utc


def test_per_airport_encoder_excludes_static_identity() -> None:
    timestamps = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i) for i in range(80)]
    one_airport = pl.DataFrame(
        {
            "airport": ["LFPG"] * 80,
            "timestamp": timestamps,
            "departure_count": [i % 7 for i in range(80)],
            "arrival_count": [(i + 2) % 5 for i in range(80)],
            "scheduled_departure_count": [(i + 1) % 8 for i in range(80)],
            "taxi_out_mean_sec": [600.0 + 3 * (i % 9) for i in range(80)],
        }
    )
    state = pl.concat(
        [one_airport, one_airport.with_columns(pl.lit("EDDF").alias("airport"))]
    )
    encoder = PerAirportPredictiveLatentEncoder.fit(
        state,
        cutoff=datetime(2025, 1, 1, 12, tzinfo=UTC),
        n_components=2,
        horizons=(1, 2),
    )
    transformed = encoder.transform(state)
    assert transformed.height == state.height
    assert transformed.get_column("airport").n_unique() == 2
    assert all(
        not feature.startswith("airport_")
        for fitted in encoder.encoders.values()
        for feature in fitted.feature_names
    )
