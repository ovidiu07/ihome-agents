```json
{
  "current_date": "2025-07-22",
  "current_day":  { "open": 65.50, "high": 68.50, "low": 65.10, "close": 66.50 },
  "weekly_forecast": { "open": 66.53, "high": 70.00, "low": 64.00, "close": 68.00 }
}
```

**SECTION 2 — step-by-step trading plan**

**Market context & key levels**

* Yesterday’s bar printed H 71.13, L 64.89, C 65.33, giving a classical pivot P 67.12, R1 69.34, S1 63.10 and a Fibonacci 0.618 retracement at 66.37   .
* Algorithm-derived support/resistance on the daily chart cluster at 62 (now good support) and the psychological band 69-70 (resistance) .
* Today’s price is consolidating just below the pivot (65.8 at 13:00 UTC) while the daily ADX remains strong (44, trending ✔) but the hourly ADX is only ≈10 (range-bound) .

**How to trade the levels**

| Level                              | Trade idea                 | Entry                       | Stop-loss\*          | First target           |
| ---------------------------------- | -------------------------- | --------------------------- | -------------------- | ---------------------- |
| 66.35 (Fib 0.618)                  | Reactive long on pull-back | 66.40-66.20                 | 65.00 (≈1 ATR below) | 67.90 (yesterday VWAP) |
| 69.30-70.00 (R1 / pattern targets) | Break-out long             | 69.40 on 15-min close above | 67.80                | 70.80 then 71.20       |
| 63.10 (S1)                         | Counter-trend scalp long   | 63.20-62.90                 | 61.80                | 64.60                  |
| 70.80 (prior swing high)           | Fade short                 | 70.75-71.10                 | 71.90                | 69.40                  |

\*Stops are volatility-adjusted with the hourly ATR ≈ 1.02 \$ .

**Active patterns & measured-moves**

* **15-min bullish triangle** projects 68.18   .
* **Hourly double-bottom** still in play with a 69.31 target and 64.45 stop.
* A minor 1-min double-top at 65.76 (target 65.51) is already mature  and merely noise inside the broader up-bias.

Pattern back-tests on this symbol show: triple-bottom win-rate 72 %, double-bottom 69 %, triangles 60 %, double-top 55 %. Weighting those probabilities, the composite directional odds for today favour bulls ≈62 %.

**Volume & momentum**

VWAP on the 5-min tape is 65.78; price is < 0.2 % above VWAP, and OBV has broken its down-trend line—bullish volume divergence   . Hourly RSI is 46-48 and rising while price makes a higher swing-low, hinting at hidden bullish divergence. No MACD bear cross on the hourly yet.

**Moving-average ribbon**
Price is < 0.1 % above the hourly 20-EMA (65.72) but still 2 % under the daily 200-EMA (≈ 67.0); ribbon remains compressed, signalling potential expansion.

**Volatility regime & forecast band**

The 14-ATR / 50-ATR ratio on the hourly sits at 0.68 → “normal-to-low” volatility; a 1 σ day therefore spans ≈ ±1.2 \$ from the mid-price. Applying this to the refined Finnhub projection (open 65.5) yields a probabilistic high 68.5 (35 % chance) and low 65.1 (40 % chance), matching Section 1. The historical 30-day MAE of Finnhub’s close forecast is 1.05 \$, with a +0.12 \$ bullish bias; our adjustment subtracts that bias.

**Risk management & sizing**

Expected reward-to-risk for the triangle breakout (entry 65.90, tgt 68.18, stop 64.63) ≈ 1.6 R. At 1 % account risk and a 1.5 \$ stop, position size = 0.67 % equity per dollar of stop. Aggregate expectancy across all qualified patterns = +0.42 R per trade.

**Bull vs. bear scenarios**

| Scenario                        | Likelihood | Key trigger                 | Target                 |
| ------------------------------- | ---------- | --------------------------- | ---------------------- |
| **Bull-case** (continuation)    | 60 %       | 66.35 hold then break 69.30 | 70.80 / 71.20          |
| **Base-case** (range)           | 30 %       | 65.10-68.50 oscillation     | Close ≈ 66.5           |
| **Bear-case** (failed triangle) | 10 %       | Hourly close < 64.90        | Slide to 63.10 / 62.00 |

---

**SECTION 3 — intraday execution plan**

*Pre-Open (13:30-14:30 UTC)*
• Bias-building only; no position.
• AI position: **flat**

*Open Range (14:30-15:00 UTC)*
• Long if first 5-min candle retests 66.35 and holds (> 50 % retrace).
• Stop 65.00; scale ⅓ at 66.90.
• AI position: **long** if triggered, else **flat**

*Mid-Morning (15:00-16:30 UTC)*
• Trail stop to hourly VWAP minus 0.5 \$.
• Add if 67.20 prints on volume > 20 % above 10-period average.
• First profit-target 68.00.
• AI position: **long/flat** depending on trigger outcome.

*Lunch (16:30-18:00 UTC)*
• No new entries; tighten stop to 66.10 (break-even + ticks).
• Exit half if price stagnates ±0.3 \$ for > 45 min.
• AI position: **long/flat**

*Afternoon push (18:00-20:00 UTC)*
• If still long and 69.30 breaks on 15-min close, trail stop to 67.80 and target 70.80.
• If price rejects 69.30 twice, flip to tactical short 70.75-71.10 with 71.90 stop, 69.40 target.
• AI position: **long** or **short** per trigger, else **flat**

*Closing hour (20:00-21:00 UTC)*
• Flatten all intraday positions 10 min before the bell unless above 70.50 with strong tape.
• AI position: **flat**

---

(ATR-based stops and all times assume the NASDAQ schedule; adjust if your venue differs.)
