from __future__ import annotations

import pandas as pd


class LabelFactory:
    """Build forward return labels for cross-sectional regression."""

    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def make_label(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        g = df.groupby("ts_code", group_keys=False)
        fwd_close = g["adj_close"].shift(-self.horizon)
        df[f"label_ret_fwd_{self.horizon}"] = (fwd_close - df["adj_close"]) / df["adj_close"]

        # remove date-wise market beta by subtracting cross-sectional mean
        col = f"label_ret_fwd_{self.horizon}"
        df[col] = df[col] - df.groupby("trade_date")[col].transform("mean")
        return df
