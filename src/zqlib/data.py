from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class AsyncQueryClient(Protocol):
    async def query_pl(self, sql: str):
        """Run SQL and return (polars_or_pandas_df, ok_flag)."""


@dataclass(slots=True)
class DuckHTTPDataSource:
    """Simple SQL data source wrapper compatible with duckhttp client."""

    client: AsyncQueryClient

    async def query_df(self, sql: str) -> pd.DataFrame:
        result, ok = await self.client.query_pl(sql)
        if not ok:
            raise RuntimeError(f"duckhttp query failed: {sql[:120]}...")
        if isinstance(result, pd.DataFrame):
            return result

        # duckhttp usually returns a polars.DataFrame from query_pl.
        to_pandas = getattr(result, "to_pandas", None)
        if callable(to_pandas):
            return to_pandas()

        raise TypeError(f"Unsupported dataframe type from duckhttp: {type(result)!r}")

    async def get_trading_calendar(self, start_date: int, end_date: int) -> pd.DataFrame:
        sql = f"""
        SELECT cal_date, pretrade_date
        FROM trade_cal
        WHERE exchange = 'SSE' AND is_open = 1
          AND cal_date BETWEEN {start_date} AND {end_date}
        ORDER BY cal_date
        """
        return await self.query_df(sql)

    async def get_universe(self, index_code: str, snapshot_month: int | None = None) -> pd.DataFrame:
        where_snapshot = (
            f"AND snapshot_month = {snapshot_month}" if snapshot_month is not None else ""
        )
        sql = f"""
        SELECT ts_code, index_code, in_date, out_date
        FROM index_member_snapshot
        WHERE index_code = '{index_code}' {where_snapshot}
        """
        return await self.query_df(sql)

    async def get_daily_panel(self, ts_codes: list[str], start_date: int, end_date: int) -> pd.DataFrame:
        if not ts_codes:
            return pd.DataFrame()

        symbols = ", ".join(f"'{x}'" for x in ts_codes)
        sql = f"""
        SELECT d.ts_code,
               d.trade_date,
               d.open,
               d.high,
               d.low,
               d.close,
               d.pre_close,
               d.change,
               d.pct_chg,
               d.vol,
               d.amount,
               a.adj_factor,
               b.turnover_rate,
               b.volume_ratio,
               b.pe_ttm,
               b.pb,
               b.ps_ttm,
               b.total_mv,
               b.circ_mv
        FROM daily d
        LEFT JOIN adj_factor a
               ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
        LEFT JOIN daily_basic b
               ON d.ts_code = b.ts_code AND d.trade_date = b.trade_date
        WHERE d.ts_code IN ({symbols})
          AND d.trade_date BETWEEN {start_date} AND {end_date}
        ORDER BY d.trade_date, d.ts_code
        """
        return await self.query_df(sql)
