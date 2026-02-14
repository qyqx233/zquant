from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureFactory:
    """Build intuitive alpha features from OHLCV-like panel."""

    @staticmethod
    def _safe_log(series: pd.Series) -> pd.Series:
        return np.log(series.clip(lower=1e-12))

    def make_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.copy()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        adj = df["adj_factor"].fillna(1.0)
        df["adj_close"] = df["close"] * adj
        df["adj_open"] = df["open"] * adj
        df["adj_high"] = df["high"] * adj
        df["adj_low"] = df["low"] * adj

        g = df.groupby("ts_code", group_keys=False)

        # Price-based momentum and reversal
        for w in (1, 5, 10, 20):
            df[f"ret_{w}"] = g["adj_close"].pct_change(w)
        df["intraday_ret"] = (df["adj_close"] - df["adj_open"]) / df["adj_open"]

        # Volatility features
        log_ret = g["adj_close"].transform(lambda x: self._safe_log(x).diff())
        df["log_ret_1"] = log_ret
        for w in (5, 10, 20):
            df[f"volatility_{w}"] = g["log_ret_1"].transform(
                lambda x: x.rolling(w, min_periods=max(3, w // 2)).std()
            )

        # Turnover, liquidity, valuation proxies
        df["amount_log"] = self._safe_log(df["amount"].replace(0, np.nan))
        df["vol_log"] = self._safe_log(df["vol"].replace(0, np.nan))
        df["turnover_rate"] = df["turnover_rate"].fillna(0.0)
        df["volume_ratio"] = df["volume_ratio"].fillna(0.0)

        df["pe_ttm_inv"] = 1.0 / df["pe_ttm"].replace(0, np.nan)
        df["pb_inv"] = 1.0 / df["pb"].replace(0, np.nan)
        df["ps_ttm_inv"] = 1.0 / df["ps_ttm"].replace(0, np.nan)

        # Cross-sectional normalize by date
        feat_cols = [
            "ret_1",
            "ret_5",
            "ret_10",
            "ret_20",
            "intraday_ret",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "amount_log",
            "vol_log",
            "turnover_rate",
            "volume_ratio",
            "pe_ttm_inv",
            "pb_inv",
            "ps_ttm_inv",
        ]

        for col in feat_cols:
            df[col] = df.groupby("trade_date")[col].transform(
                lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
            )

        return df
