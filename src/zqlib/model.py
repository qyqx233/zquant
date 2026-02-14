from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd


@dataclass(slots=True)
class LightGBMAlphaModel:
    """LightGBM regressor for cross-sectional alpha prediction."""

    params: dict
    num_boost_round: int = 300
    early_stopping_rounds: int = 50

    def __post_init__(self):
        self.booster: lgb.Booster | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
    ) -> lgb.Booster:
        dtrain = lgb.Dataset(train_df[feature_cols], label=train_df[label_col])
        dvalid = lgb.Dataset(valid_df[feature_cols], label=valid_df[label_col], reference=dtrain)

        self.booster = lgb.train(
            params=self.params,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            num_boost_round=self.num_boost_round,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(50)],
        )
        return self.booster

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("Model has not been fitted yet.")
        pred = self.booster.predict(df[feature_cols])
        return pd.Series(pred, index=df.index, name="score")

    @staticmethod
    def evaluate_ic(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
        """Information Coefficient (cross-sectional Spearman) by date."""
        ic = (
            df.groupby("trade_date")[[score_col, label_col]]
            .corr(method="spearman")
            .unstack()
            .iloc[:, 1]
            .rename("ic")
            .reset_index()
        )
        return ic
