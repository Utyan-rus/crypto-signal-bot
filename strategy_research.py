"""Compare three OHLCV strategy families with strict chronological holdout on Lighter.
Paper research only. No private endpoints or orders. Signals use closed candles and entries
occur at the next candle open; estimated fees and slippage are charged on both sides.
"""
import argparse
import csv
import json
import math
import os
import time

from backtest import fetch_candles, get_markets, ema, rsi_series

FEE = 0.0005
SLIPPAGE = 0.0005
START_BALANCE = 5.0
RISK_FRACTION = 0.005
LEVERAGE = 3.0
MIN_NOTIONAL = 1.0
STRATEGIES = ("donchian_breakout", "volume_momentum", "rsi_bollinger", "guarded_baseline")


def atr_series(candles, period=14):
    out = [None] * len(candles)
    trs = []
    for i, bar in enumerate(candles):
        prev = candles[i-1]["c"] if i else bar["c"]
        trs.append(max(bar["h"]-bar["l"], abs(bar["h"]-prev), abs(bar["l"]-prev)))
        if i >= period:
            out[i] = sum(trs[i-period+1:i+1]) / period
    return out


def rolling_mean(values, period):
    out = [None] * len(values)
    for i in range(period-1, len(values)):
        out[i] = sum(values[i-period+1:i+1]) / period
    return out


def rolling_std(values, period):
    out = [None] * len(values)
    for i in range(period-1, len(values)):
        chunk = values[i-period+1:i+1]
        mean = sum(chunk)/period
        out[i] = math.sqrt(sum((x-mean)**2 for x in chunk)/period)
    return out


def signal_map(candles, strategy):
    """Map candle timestamp -> (side, stop distance, target R)."""
    close = [x["c"] for x in candles]
    high = [x["h"] for x in candles]
    low = [x["l"] for x in candles]
    volume = [x.get("v", 0.0) for x in candles]
    atr = atr_series(candles)
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    rsi = rsi_series(close)
    vmean = rolling_mean(volume, 20)
    mid, std = rolling_mean(close, 20), rolling_std(close, 20)
    signals = {}
    for i, bar in enumerate(candles):
        if atr[i] is None or atr[i] <= 0:
            continue
        side, rr = "NEUTRAL", 2.0
        if strategy == "donchian_breakout":
            # Exclude current bar from channel so the breakout is not self-referential.
            if i >= 56 and e200[i] is not None:
                upper = max(high[i-20:i])
                lower = min(low[i-20:i])
                vol_ok = vmean[i] is not None and volume[i] >= 0.8*vmean[i]
                if close[i] > upper and close[i] > e200[i] and vol_ok:
                    side = "LONG"
                elif close[i] < lower and close[i] < e200[i] and vol_ok:
                    side = "SHORT"
                rr = 2.0
        elif strategy == "volume_momentum":
            if i >= 51 and e20[i] is not None and e50[i] is not None and vmean[i] and vmean[i] > 0:
                vol_ok = volume[i] >= 1.2*vmean[i]
                if close[i] > e20[i] > e50[i] and close[i] > close[i-3] and vol_ok:
                    side = "LONG"
                elif close[i] < e20[i] < e50[i] and close[i] < close[i-3] and vol_ok:
                    side = "SHORT"
                rr = 2.0
        elif strategy == "rsi_bollinger":
            if mid[i] is not None and std[i] is not None and rsi[i] is not None:
                upper, lower = mid[i]+2*std[i], mid[i]-2*std[i]
                # Reversion only when price is outside the band and RSI confirms exhaustion.
                if close[i] < lower and rsi[i] <= 30:
                    side = "LONG"
                elif close[i] > upper and rsi[i] >= 70:
                    side = "SHORT"
                rr = 1.5
        elif strategy == "guarded_baseline":
            # A lightweight OHLCV approximation for a comparison baseline; live strategy
            # also uses MACD/ADX and 15m confirmation, so this is not an exact clone.
            if e20[i] is not None and e50[i] is not None and e200[i] is not None and rsi[i] is not None and vmean[i]:
                vol_ok = volume[i] >= 0.8*vmean[i]
                if close[i] > e200[i] and e20[i] > e50[i] > e200[i] and 52 <= rsi[i] <= 68 and vol_ok:
                    side = "LONG"
                elif close[i] < e200[i] and e20[i] < e50[i] < e200[i] and 32 <= rsi[i] <= 48 and vol_ok:
                    side = "SHORT"
                rr = 2.0
        if side != "NEUTRAL":
            signals[bar["t"]] = (side, max(atr[i]*1.5, close[i]*0.002), rr)
    return signals


def simulate(symbol, candles, strategy):
    signals = signal_map(candles, strategy)
    balance = peak = START_BALANCE
    max_dd = 0.0
    trades = []
    pos = None
    for i in range(1, len(candles)):
        prev, bar = candles[i-1], candles[i]
        if pos:
            exit_price, reason = None, None
            if pos["side"] == "LONG":
                if bar["l"] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "SL"
                elif bar["h"] >= pos["target"]:
                    exit_price, reason = pos["target"], "TP"
            else:
                if bar["h"] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "SL"
                elif bar["l"] <= pos["target"]:
                    exit_price, reason = pos["target"], "TP"
            if exit_price is not None:
                exit_price *= (1-SLIPPAGE if pos["side"] == "LONG" else 1+SLIPPAGE)
                gross = pos["qty"]*(exit_price-pos["entry"])*(1 if pos["side"]=="LONG" else -1)
                pnl = gross - abs(pos["qty"]*exit_price)*FEE - pos["entry_fee"]
                balance += pnl
                trades.append({"symbol":symbol,"strategy":strategy,"side":pos["side"],
                               "entry_time":pos["entry_time"],"exit_time":bar["t"],
                               "reason":reason,"pnl":pnl})
                peak=max(peak,balance)
                max_dd=max(max_dd,(peak-balance)/peak if peak>0 else 1.0)
                pos=None
        # Signals known only after prev candle closes; enter at current open.
        sig = signals.get(prev["t"])
        if pos is None and sig and balance > 0:
            side, stop_distance, rr = sig
            entry = bar["o"]*(1+SLIPPAGE if side=="LONG" else 1-SLIPPAGE)
            qty = min(balance*RISK_FRACTION/(stop_distance+entry*2*(FEE+SLIPPAGE)),
                      balance*LEVERAGE/entry)
            if qty <= 0 or qty*entry < MIN_NOTIONAL:
                continue
            stop = entry-stop_distance if side=="LONG" else entry+stop_distance
            target = entry+stop_distance*rr if side=="LONG" else entry-stop_distance*rr
            pos={"side":side,"entry":entry,"qty":qty,"stop":stop,"target":target,
                 "entry_fee":qty*entry*FEE,"entry_time":bar["t"]}
            # Conservative same-bar handling: if both levels touch, stop wins.
            reason = None
            if side=="LONG":
                if bar["l"] <= stop: exit_price,reason=stop,"SL"
                elif bar["h"] >= target: exit_price,reason=target,"TP"
                else: continue
            else:
                if bar["h"] >= stop: exit_price,reason=stop,"SL"
                elif bar["l"] <= target: exit_price,reason=target,"TP"
                else: continue
            exit_price *= (1-SLIPPAGE if side=="LONG" else 1+SLIPPAGE)
            pnl=qty*(exit_price-entry)*(1 if side=="LONG" else -1)-abs(qty*exit_price)*FEE-pos["entry_fee"]
            balance+=pnl
            trades.append({"symbol":symbol,"strategy":strategy,"side":side,"entry_time":bar["t"],
                           "exit_time":bar["t"],"reason":reason,"pnl":pnl})
            peak=max(peak,balance)
            max_dd=max(max_dd,(peak-balance)/peak if peak>0 else 1.0)
            pos=None
    if pos:
        bar=candles[-1]
        exit_price=bar["c"]*(1-SLIPPAGE if pos["side"]=="LONG" else 1+SLIPPAGE)
        pnl=pos["qty"]*(exit_price-pos["entry"])*(1 if pos["side"]=="LONG" else -1)-abs(pos["qty"]*exit_price)*FEE-pos["entry_fee"]
        balance+=pnl
        trades.append({"symbol":symbol,"strategy":strategy,"side":pos["side"],"entry_time":pos["entry_time"],
                       "exit_time":bar["t"],"reason":"END","pnl":pnl})
        peak=max(peak,balance)
        max_dd=max(max_dd,(peak-balance)/peak if peak>0 else 1.0)
    wins=sum(t["pnl"]>0 for t in trades)
    gross_profit=sum(t["pnl"] for t in trades if t["pnl"]>0)
    gross_loss=-sum(t["pnl"] for t in trades if t["pnl"]<0)
    return {"symbol":symbol,"strategy":strategy,"trades":len(trades),"wins":wins,
            "win_rate_pct":round(100*wins/len(trades),2) if trades else 0,
            "start_balance":START_BALANCE,"end_balance":round(balance,6),
            "return_pct":round(100*(balance/START_BALANCE-1),3),
            "max_drawdown_pct":round(max_dd*100,3),
            "profit_factor":round(gross_profit/gross_loss,3) if gross_loss else (None if not gross_profit else 999.0),
            "fees_per_side_pct":FEE*100,"slippage_per_side_pct":SLIPPAGE*100,
            "funding":"not included; historical funding feed unavailable",
            "status":"tested" if len(trades)>=5 else "low_sample"}


def split_history(candles, test_days=30):
    if not candles: return None
    cutoff=candles[-1]["t"]-test_days*86400
    train=[c for c in candles if c["t"]<cutoff]
    test=[c for c in candles if c["t"]>=cutoff]
    if len(train)<220*3 or len(test)<220: return None
    return train,test


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--days",type=int,default=90)
    p.add_argument("--max-markets",type=int,default=5)
    p.add_argument("--out",default="strategy-research-results")
    args=p.parse_args()
    if args.days!=90: p.error("Requires 90 days (60 train + 30 out-of-sample).")
    os.makedirs(args.out,exist_ok=True)
    markets=get_markets()
    if args.max_markets>0: markets=markets[:args.max_markets]
    end=int(time.time()); start=end-90*86400
    results=[]; failures=[]; trades_all=[]
    print(f"Comparing {len(STRATEGIES)} strategies on {len(markets)} markets: 60d train / 30d holdout.")
    for n,market in enumerate(markets,1):
        symbol=market["symbol"]
        print(f"[{n}/{len(markets)}] {symbol}")
        try:
            c5=fetch_candles(market["id"],"5m",start,end)
            expected=90*288
            if len(c5)<expected*0.9:
                results.append({"symbol":symbol,"strategy":"ALL","status":"incomplete_history","bars_5m":len(c5)})
                continue
            split=split_history(c5)
            if not split:
                results.append({"symbol":symbol,"strategy":"ALL","status":"insufficient_split_history"})
                continue
            train,test=split
            for strategy in STRATEGIES:
                train_result=simulate(symbol,train,strategy)
                test_result=simulate(symbol,test,strategy)
                row={"symbol":symbol,"strategy":strategy,"status":test_result["status"],
                     "train_trades":train_result["trades"],"train_return_pct":train_result["return_pct"],
                     "train_max_drawdown_pct":train_result["max_drawdown_pct"],
                     "test_trades":test_result["trades"],"test_wins":test_result["wins"],
                     "test_win_rate_pct":test_result["win_rate_pct"],"test_return_pct":test_result["return_pct"],
                     "test_max_drawdown_pct":test_result["max_drawdown_pct"],"test_profit_factor":test_result["profit_factor"],
                     "test_start_balance":START_BALANCE,"test_end_balance":test_result["end_balance"],
                     "fees_per_side_pct":FEE*100,"slippage_per_side_pct":SLIPPAGE*100,
                     "funding":"not included"}
                results.append(row)
                trades_all.extend(simulate(symbol,test,strategy).get("trades_detail",[]))
                print(f"  {strategy}: train {train_result['return_pct']:+.2f}%, OOS {test_result['return_pct']:+.2f}% ({test_result['trades']} trades, DD {test_result['max_drawdown_pct']:.2f}%)")
        except Exception as exc:
            failures.append({"symbol":symbol,"error":type(exc).__name__+": "+str(exc)[:180]})
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:180]}")
        time.sleep(0.15)
    fields=sorted({k for row in results for k in row})
    with open(os.path.join(args.out,"walk_forward_summary.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    with open(os.path.join(args.out,"candidate_details.json"),"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    with open(os.path.join(args.out,"failures.json"),"w",encoding="utf-8") as f:
        json.dump(failures,f,ensure_ascii=False,indent=2)
    tested=[r for r in results if r.get("test_trades",0)>=5]
    profitable=[r for r in tested if r["test_return_pct"]>0]
    print(f"Finished: {len(tested)} market/strategy holdouts; {len(profitable)} positive after estimated fees/slippage; {len(failures)} API failures.")
    print("Research only. Funding excluded; guarded baseline is an OHLCV approximation, not an exact live-strategy clone. No orders placed.")
if __name__=="__main__": main()
