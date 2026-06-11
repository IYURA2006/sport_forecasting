# World Cup 2026 forecasting

Dixon-Coles scoreline model for international football, evaluated against
de-vigged bookmaker probabilities rather than raw accuracy. The goal is daily,
git-frozen probabilistic forecasts for the 2026 World Cup with honest
uncertainty intervals, plus a leakage-proof walk-forward backtest over the
2014/2018/2022 tournaments.

Status: early. The data layer is in (typed parquet snapshots with
point-in-time timestamps); model, backtest and tournament simulator are next.

## Setup

```
make venv
make data
make test
```

Results data: martj42's [International football results from 1872](https://github.com/martj42/international_results),
dropped into `data/raw/results.csv` with the 2026 group-stage schedule appended
as NA-score rows.
