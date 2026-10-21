"""Apply the THY LIRF January 2026 substitute-regime rule (EXP-050) to a candidate file.

Usage: uv run python scripts/protocol/apply_thy.py <in.parquet> <out-name> <latent final.parquet>
Writes data/submissions/candidate_<out-name>.parquet. The non-fill part of the rule uses the given
latent's nn_pred (the arrival-token latent in the freeze candidate).
"""
import sys

import numpy as np
import polars as pl

from prc.data import load_ranking
from prc.models.postrules import apply_thy, thy_regime


def main(src: str, out: str, latent: str) -> None:
    tmpl = pl.read_parquet("data/raw/submitting.parquet")
    sub = pl.read_parquet(src)
    if not sub["MVT_ID_mvt"].equals(tmpl["MVT_ID_mvt"]):
        raise ValueError("ids or order differ from the template")
    thy = thy_regime(load_ranking(), None)
    nn = pl.read_parquet(latent, columns=["MVT_ID_mvt", "nn_pred"]).join(thy.select("MVT_ID_mvt"), on="MVT_ID_mvt")
    est = dict(zip(nn["MVT_ID_mvt"].to_list(), nn["nn_pred"].to_list()))
    pred = sub["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    new = np.clip(np.rint(apply_thy(pred, sub["MVT_ID_mvt"].cast(pl.Int64).to_numpy(), thy, est)), 0, None).astype(np.int32)
    res = sub.with_columns(pl.Series("TAXITIME_SEC_mvt", new))
    res.write_parquet(f"data/submissions/candidate_{out}.parquet")
    print(f"candidate_{out}.parquet: {res.height} rows, {int((new != pred).sum())} THY rows changed")


if __name__ == "__main__":
    main(*sys.argv[1:4])
