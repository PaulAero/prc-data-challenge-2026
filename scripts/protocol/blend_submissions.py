"""Average candidate submission files (same template order) into one blended candidate.

Usage: uv run python scripts/protocol/blend_submissions.py <out-name> <cand1.parquet> <cand2.parquet> [...]
Writes data/submissions/candidate_<out-name>.parquet and a .check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

TEMPLATE = Path("data/raw/submitting.parquet")


def main(out: str, files: list[str]) -> None:
    tmpl = pl.read_parquet(TEMPLATE).select("MVT_ID_mvt")
    preds = []
    for f in files:
        c = pl.read_parquet(f)
        if not c["MVT_ID_mvt"].equals(tmpl["MVT_ID_mvt"]):
            raise ValueError(f"{f}: ids or order differ from the template")
        preds.append(c["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy())
    blend = np.rint(np.mean(preds, axis=0)).astype(np.int32)
    sub = tmpl.with_columns(pl.Series("TAXITIME_SEC_mvt", blend))
    path = Path("data/submissions") / f"candidate_{out}.parquet"
    sub.write_parquet(path)
    check = {"file": str(path), "blend_of": files, "rows": sub.height, "nulls": int(sub["TAXITIME_SEC_mvt"].null_count()),
             "dtype": str(sub.schema), "pred_min": int(blend.min()), "pred_max": int(blend.max()),
             "pairwise_mean_abs_diff": float(np.mean(np.abs(preds[0] - preds[-1])))}
    Path(str(path).replace(".parquet", ".check.json")).write_text(json.dumps(check, indent=1))
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
