# zqlib (qlib 风格轻量复刻)

一个面向 **A 股横截面预测** 的最小可用框架，目标是：

- 不依赖 qlib 主框架；
- 结构上保留 qlib 的核心思想（Data -> Feature -> Label -> Dataset -> Model）；
- 先支持 LightGBM，方便你快速迭代想法。

## 目录结构

```text
src/zqlib/
  config.py    # 实验配置
  data.py      # duckhttp 数据读取封装
  features.py  # 因子/特征构造
  label.py     # 监督标签构造
  dataset.py   # 数据集拼装 + 训练/验证/测试切分
  model.py     # LightGBM 训练、预测、IC 评估
examples/
  train_lgbm.py
```

## 安装

```bash
pip install -e .
```

## 与你的 duckhttp 对接

你已有调用方式：

```python
sql = f"""
SELECT d.trade_date, d.open, d.high, d.low, d.close, d.vol, d.amount,
       a.adj_factor
FROM daily d
LEFT JOIN adj_factor a ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
WHERE d.ts_code = '{ts_code}' {date_filter}
ORDER BY d.trade_date
"""
df, ok = await dc.query_pl(sql)
```

`zqlib.data.DuckHTTPDataSource` 已按同样接口封装（`query_pl(sql) -> (df, ok)`），只需要把你的真实 `dc` client 注入进去即可。

## 最小训练流程

1. 构建 `ExperimentConfig`（时间范围、标签 horizon、指数成分股范围）；
2. `CrossSectionalDatasetBuilder.build()` 拉数据 + 构造特征 + 构造标签 + 切分 train/valid/test；
3. `LightGBMAlphaModel.fit()` 训练；
4. `predict()` 得到 score，`evaluate_ic()` 计算横截面 IC。

参考：`examples/train_lgbm.py`

## 当前内置特征（可扩展）

- 收益率动量：`ret_1/5/10/20`
- 日内收益：`intraday_ret`
- 波动率：`volatility_5/10/20`
- 流动性：`amount_log`, `vol_log`, `turnover_rate`, `volume_ratio`
- 估值倒数：`pe_ttm_inv`, `pb_inv`, `ps_ttm_inv`
- 按交易日做横截面 z-score 标准化

## 标签定义

- 默认 `horizon=5`：`label_ret_fwd_5 = future_adj_close / adj_close - 1`
- 再减去当日横截面均值，得到 market-neutral label

## 下一步建议

- 增加行业中性化（基于 `industry_classify`）；
- 增加停牌/涨跌停过滤与交易可行性约束；
- 增加更完整的回测接口（分组收益、换手、IR）；
- 增加模型注册机制（XGBoost/CatBoost/线性模型）。
