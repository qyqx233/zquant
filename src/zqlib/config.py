from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UniverseConfig:
    """Universe selection settings."""

    index_code: str = "000300.SH"
    snapshot_month: int | None = None
    include_st: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    """Core experiment setup for cross-sectional prediction."""

    start_date: int
    end_date: int
    train_end_date: int
    valid_end_date: int
    label_horizon: int = 5
    min_history: int = 60
    universe: UniverseConfig = field(default_factory=UniverseConfig)
