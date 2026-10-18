# Backtest methodology audit (2026-10-18)

## Findings before promoting any strategy

1. **Drawdown is close-to-close/trade-exit based, not mark-to-market.** Both research simulators update peak equity and drawdown after realized exits. An open position's adverse excursion is not included. Reported max drawdown therefore understates path risk and must not be used as a live-risk estimate.
2. **Indicator warm-up is reset at the train/test boundary.** Research calls the signal generator separately on the train and test slices. EMA/rolling indicators in the holdout do not inherit the preceding 60 days. This discards early holdout signals and can change indicator values relative to continuous operation. A corrected test should calculate indicators on context candles but prohibit entries before the holdout timestamp.
3. **Funding is omitted.** Results are gross of perpetual funding; long/short funding can materially change net expectancy.
4. **Small-account constraints are simplified.** A $5 balance with a $1 minimum notional, leverage cap and fixed risk fraction may have discrete sizing/rounding effects not represented by the simulator. Exchange fees, actual fills, tick/lot sizes and liquidation rules should be verified before deployment.
5. **Multiple-comparison risk is high.** Many strategy/filter/entry combinations are tested on the same short holdout. The best observed row is subject to selection bias. Keep a final untouched period or use nested walk-forward evaluation, and report per-market results and confidence intervals.
6. **Baseline is not an exact live-strategy clone.** The OHLCV guarded baseline omits parts of the live strategy such as MACD/ADX and 15-minute confirmation. It is only a comparator.

## Promotion gates

Do not enable live trading based on a positive row alone. Require:
- continuous indicator warm-up with strictly chronological entries;
- mark-to-market equity and drawdown, conservative intrabar stop handling;
- fee/slippage stress tests and funding where data is available;
- more than one out-of-sample window and broad market coverage;
- positive net expectancy after costs, sufficient trade count, and acceptable drawdown;
- paper-forward validation before considering any live use.

This audit does not change the bot's live strategy and places no orders.
