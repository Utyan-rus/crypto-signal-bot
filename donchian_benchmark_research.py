"""Paired Donchian 20/10 vs buy-and-hold on identical market/OOS windows. Paper-only."""
import argparse, csv, os, statistics, time
from backtest import fetch_candles, get_markets
from funding_aware_research import fetch_funding_rates, make_oos_windows, simulate_with_funding
from hourly_candidate_research import FEE, SLIPPAGE

MIN_QUOTE_NOTIONAL_USD = 10.0

def max_drawdown_pct(equity):
    peak = equity[0] if equity else 0.0
    result = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            result = max(result, (peak-value)/peak)
    return round(100*result, 3)

def buy_hold(symbol, candles, start, end, funding, balance):
    bars = [b for b in candles if start <= b["t"] < end]
    if len(bars) < 2:
        return {"symbol":symbol,"strategy":"buy_hold_long_1x","status":"low_sample","trades":0,"start_balance":balance}
    entry = bars[0]["o"]*(1+SLIPPAGE)
    qty = balance/(entry*(1+FEE))
    entry_fee = qty*entry*FEE
    funding_cost, charged = 0.0, 0
    cash_after_entry = balance-qty*entry-entry_fee
    equity = [balance]
    for bar in bars[1:]:
        if bar["t"] in funding:
            funding_cost += qty*bar["c"]*funding[bar["t"]]
            charged += 1
        mark = bar["c"]*(1-SLIPPAGE)
        equity.append(cash_after_entry+qty*mark-qty*mark*FEE-funding_cost)
    exit_price = bars[-1]["c"]*(1-SLIPPAGE)
    end_balance = balance+qty*(exit_price-entry)-entry_fee-qty*exit_price*FEE-funding_cost
    notional = qty*entry
    eligible = notional >= MIN_QUOTE_NOTIONAL_USD
    return {"symbol":symbol,"strategy":"buy_hold_long_1x","status":"tested","trades":1,
      "start_balance":balance,"end_balance":round(end_balance,6),
      "return_pct":round(100*(end_balance/balance-1),3),
      "max_drawdown_mtm_pct":max_drawdown_pct(equity),"funding_events_charged":charged,
      "funding_net_cost_usd":round(funding_cost,8),"min_quote_notional_usd":round(notional,6),
      "min_quote_check_pass":eligible,
      "execution_feasibility":"quote_minimum_pass_other_constraints_unverified" if eligible else "below_min_quote"}

def paired_rows(strategy, benchmark, window):
    sret, bret = float(strategy.get("return_pct",0) or 0), float(benchmark.get("return_pct",0) or 0)
    trades = int(strategy.get("trades",0) or 0)
    strategy_ok = strategy.get("status") == "tested" and trades >= 5
    benchmark_ok = benchmark.get("status") == "tested" and bool(benchmark.get("min_quote_check_pass"))
    paired = strategy_ok and benchmark_ok
    return {"symbol":strategy["symbol"],"window":window,"start_balance":strategy["start_balance"],
      "strategy":"donchian_20_10","strategy_status":strategy.get("status",""),"strategy_trades":trades,
      "strategy_return_pct":sret,"strategy_mtm_drawdown_pct":strategy.get("max_drawdown_mtm_pct",""),
      "strategy_profit_factor":strategy.get("profit_factor",""),
      "strategy_funding_cost_usd":strategy.get("funding_net_cost_usd",""),
      "benchmark_return_pct":bret,"benchmark_mtm_drawdown_pct":benchmark.get("max_drawdown_mtm_pct",""),
      "benchmark_funding_cost_usd":benchmark.get("funding_net_cost_usd",""),
      "benchmark_notional_usd":benchmark.get("min_quote_notional_usd",""),
      "benchmark_min_notional_pass":benchmark.get("min_quote_check_pass",False),
      "strategy_eligible_5plus_trades":strategy_ok,"benchmark_eligible":benchmark_ok,
      "paired_evaluable":paired,"excess_return_pp":round(sret-bret,3) if paired else "",
      "strategy_beats_benchmark":sret>bret if paired else False}

def summarize(rows):
    out=[]
    for balance in sorted({float(r["start_balance"]) for r in rows}):
        subset=[r for r in rows if float(r["start_balance"])==balance]
        for scope,data in (("all_rows",subset),("paired_evaluable",[r for r in subset if r["paired_evaluable"]])):
            pairs=[r for r in data if r["paired_evaluable"]]
            excess=[float(r["excess_return_pp"]) for r in pairs]
            sr=[float(r["strategy_return_pct"]) for r in data]
            br=[float(r["benchmark_return_pct"]) for r in data]
            sdd=[float(r["strategy_mtm_drawdown_pct"]) for r in pairs if r["strategy_mtm_drawdown_pct"] not in ("",None)]
            bdd=[float(r["benchmark_mtm_drawdown_pct"]) for r in pairs if r["benchmark_mtm_drawdown_pct"] not in ("",None)]
            out.append({"start_balance":balance,"scope":scope,"rows":len(data),"paired_count":len(pairs),
              "strategy_mean_return_pct":round(statistics.mean(sr),3) if sr else "",
              "strategy_median_return_pct":round(statistics.median(sr),3) if sr else "",
              "benchmark_mean_return_pct":round(statistics.mean(br),3) if br else "",
              "benchmark_median_return_pct":round(statistics.median(br),3) if br else "",
              "mean_excess_return_pp":round(statistics.mean(excess),3) if excess else "",
              "median_excess_return_pp":round(statistics.median(excess),3) if excess else "",
              "strategy_wins":sum(bool(r["strategy_beats_benchmark"]) for r in pairs),
              "mean_strategy_mtm_dd_pct":round(statistics.mean(sdd),3) if sdd else "",
              "mean_benchmark_mtm_dd_pct":round(statistics.mean(bdd),3) if bdd else ""})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--days",type=int,default=365); p.add_argument("--windows",type=int,default=8)
    p.add_argument("--window-days",type=int,default=30); p.add_argument("--max-markets",type=int,default=10)
    p.add_argument("--balances",default="5,50,100")
    p.add_argument("--end-timestamp",type=int,default=0)
    p.add_argument("--out",default="donchian-benchmark-results")
    args=p.parse_args()
    if args.windows<1 or args.window_days<1: p.error("--windows and --window-days must be positive")
    end=args.end_timestamp or int(time.time()); start=end-args.days*86400
    balances=[float(x.strip()) for x in args.balances.split(",") if x.strip()]
    if not balances or any(x<=0 for x in balances): p.error("--balances must contain positive values")
    os.makedirs(args.out,exist_ok=True)
    windows=make_oos_windows(end,count=args.windows,window_days=args.window_days)
    markets=get_markets()
    if args.max_markets>0: markets=markets[:args.max_markets]
    rows,errors=[],[]
    for market in markets:
        symbol,mid=market["symbol"],market["id"]
        try:
            candles=fetch_candles(mid,"1h",start,end); funding=fetch_funding_rates(mid,start,end)
            if len(candles)<args.days*24*0.8:
                errors.append({"symbol":symbol,"error":f"incomplete_candles:{len(candles)}"}); continue
            if not funding: errors.append({"symbol":symbol,"error":"missing_funding_history"}); continue
            for window,lo,hi in windows:
                for balance in balances:
                    strat=simulate_with_funding(symbol,candles,"donchian_20_10",lo,hi,funding,start_balance=balance)
                    bench=buy_hold(symbol,candles,lo,hi,funding,balance)
                    row=paired_rows(strat,bench,window)
                    row.update({"funding_coverage_pct":round(100*sum(lo<=t<hi for t in funding)/max(1,(hi-lo)//3600),2),
                      "fees_per_side_pct":FEE*100,"slippage_per_side_pct":SLIPPAGE*100,"shared_end_timestamp":end})
                    rows.append(row)
        except Exception as exc: errors.append({"symbol":symbol,"error":type(exc).__name__+": "+str(exc)[:200]})
        time.sleep(0.15)
    def write(path,data):
        with open(os.path.join(args.out,path),"w",newline="",encoding="utf-8") as f:
            fields=sorted({k for r in data for k in r})
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    write("donchian_benchmark_pairs.csv",rows)
    summaries=summarize(rows); write("donchian_benchmark_summary.csv",summaries)
    write("errors.csv",errors)
    paired=[r for r in rows if r["paired_evaluable"]]
    print(f"Rows={len(rows)} paired_evaluable={len(paired)} errors={len(errors)} shared_end_timestamp={end}. Paper only.")
    for balance in balances:
        data=[r for r in paired if float(r["start_balance"])==balance]
        if data:
            vals=[float(r["excess_return_pp"]) for r in data]
            print(f"balance={balance:g}: paired={len(data)} mean_excess={statistics.mean(vals):+.3f}pp median_excess={statistics.median(vals):+.3f}pp wins={sum(r['strategy_beats_benchmark'] for r in data)}/{len(data)}")
        else: print(f"balance={balance:g}: no executable paired samples")

if __name__=="__main__": main()
