from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ExperimentConfig
from .data import DuckHTTPDataSource
from .features import FeatureFactory
from .label import LabelFactory


@dataclass(slots=True)
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    label_col: str


class CrossSectionalDatasetBuilder:
    """Assemble qlib-style panel dataset with date-based train/valid/test split."""

    def __init__(
        self,
        data_source: DuckHTTPDataSource,
        feature_factory: FeatureFactory,
        label_factory: LabelFactory,
    ):
        self.data_source = data_source
        self.feature_factory = feature_factory
        self.label_factory = label_factory

    async def build(self, config: ExperimentConfig) -> DatasetBundle:
        members = await self.data_source.get_universe(
            index_code=config.universe.index_code,
            snapshot_month=config.universe.snapshot_month,
        )
        ts_codes = sorted(members["ts_code"].dropna().unique().tolist())

        panel = await self.data_source.get_daily_panel(
            ts_codes=ts_codes,
            start_date=config.start_date,
            end_date=config.end_date,
        )

        panel = self.feature_factory.make_features(panel)
        panel = self.label_factory.make_label(panel)

        label_col = f"label_ret_fwd_{config.label_horizon}"
        feature_cols = [
            x
            for x in panel.columns
            if x
            not in {
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "change",
                "pct_chg",
                "vol",
                "amount",
                "adj_factor",
                "adj_open",
                "adj_high",
                "adj_low",
                "adj_close",
                label_col,
            }
        ]

        core = panel[["trade_date", "ts_code", *feature_cols, label_col]].dropna().reset_index(drop=True)

        train = core[core["trade_date"] <= config.train_end_date].reset_index(drop=True)
        valid = core[
            (core["trade_date"] > config.train_end_date)
            & (core["trade_date"] <= config.valid_end_date)
        ].reset_index(drop=True)
        test = core[core["trade_date"] > config.valid_end_date].reset_index(drop=True)

        return DatasetBundle(
            train=train,
            valid=valid,
            test=test,
            feature_cols=feature_cols,
            label_col=label_col,
        )
