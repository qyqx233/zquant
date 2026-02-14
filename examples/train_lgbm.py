from __future__ import annotations

import asyncio
from dataclasses import dataclass

from zqlib import (
    CrossSectionalDatasetBuilder,
    DuckHTTPDataSource,
    ExperimentConfig,
    FeatureFactory,
    LabelFactory,
    LightGBMAlphaModel,
    UniverseConfig,
)


@dataclass
class DummyDuckHTTPClient:
    """
    Replace this with your real duckhttp client.

    Required interface:
        async def query_pl(self, sql: str) -> tuple[df, bool]
    """

    async def query_pl(self, sql: str):
        raise NotImplementedError("Please inject a real duckhttp client implementation")


async def main():
    duck_client = DummyDuckHTTPClient()
    data_source = DuckHTTPDataSource(duck_client)

    config = ExperimentConfig(
        start_date=20180101,
        end_date=20241231,
        train_end_date=20221231,
        valid_end_date=20231231,
        label_horizon=5,
        universe=UniverseConfig(index_code="000300.SH", snapshot_month=202412),
    )

    builder = CrossSectionalDatasetBuilder(
        data_source=data_source,
        feature_factory=FeatureFactory(),
        label_factory=LabelFactory(horizon=config.label_horizon),
    )
    bundle = await builder.build(config)

    model = LightGBMAlphaModel(
        params={
            "objective": "regression",
            "metric": ["l2", "l1"],
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "seed": 42,
            "verbosity": -1,
        },
        num_boost_round=500,
        early_stopping_rounds=50,
    )

    model.fit(
        train_df=bundle.train,
        valid_df=bundle.valid,
        feature_cols=bundle.feature_cols,
        label_col=bundle.label_col,
    )

    test = bundle.test.copy()
    test["score"] = model.predict(test, bundle.feature_cols)
    ic_df = model.evaluate_ic(test, score_col="score", label_col=bundle.label_col)

    print("test rows:", len(test))
    print("mean IC:", ic_df["ic"].mean())


if __name__ == "__main__":
    asyncio.run(main())
