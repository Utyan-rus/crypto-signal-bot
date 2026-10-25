import unittest
from donchian_benchmark_research import paired_rows, summarize, buy_hold

class DonchianBenchmarkTests(unittest.TestCase):
    def test_pair_requires_trade_and_notional_eligibility(self):
        s={"symbol":"BTC","status":"tested","trades":4,"start_balance":5,"return_pct":10,"max_drawdown_mtm_pct":1}
        b={"symbol":"BTC","status":"tested","trades":1,"return_pct":-10,"min_quote_check_pass":False,"min_quote_notional_usd":5}
        row=paired_rows(s,b,"oos_01")
        self.assertFalse(row["paired_evaluable"]); self.assertEqual(row["excess_return_pp"],"")
        self.assertFalse(row["strategy_beats_benchmark"])
    def test_valid_pair_excess_and_summary(self):
        s={"symbol":"ETH","status":"tested","trades":8,"start_balance":50,"return_pct":3,"max_drawdown_mtm_pct":4,"profit_factor":1.2}
        b={"symbol":"ETH","status":"tested","trades":1,"return_pct":1,"min_quote_check_pass":True,"min_quote_notional_usd":49,"max_drawdown_mtm_pct":6}
        row=paired_rows(s,b,"oos_01")
        self.assertTrue(row["paired_evaluable"]); self.assertEqual(row["excess_return_pp"],2)
        self.assertEqual(summarize([row])[1]["mean_excess_return_pp"],2)
    def test_small_balance_below_notional(self):
        c=[{"t":0,"o":100,"h":101,"l":99,"c":100},{"t":3600,"o":100,"h":101,"l":99,"c":100}]
        r=buy_hold("BTC",c,0,7200,{},5)
        self.assertFalse(r["min_quote_check_pass"]); self.assertEqual(r["execution_feasibility"],"below_min_quote")

if __name__=="__main__": unittest.main()
