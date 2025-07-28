### SECTION 1 — concise forecast JSON

```json
{
  "current_date": "2025-07-28",
  "current_time": "2025-07-28T14:05:00Z",
  "today_forecast": { "open": 174.24, "high": 176.80, "low": 172.80, "close": 175.50 },
  "weekly_forecast": { "open": 166.47, "high": 178.00, "low": 163.80, "close": 176.00 }
}
```

### SECTION 2 — expert trading plan

**1. Updated multi-TF snapshot**

*Price action.* The last 1-min close printed **174.24** after a relentless melt-up from the midday pivot at 169.71; that leg cleared the prior resistance shelf at 172.98 and established fresh intraday highs .

*Trend strength.* One-minute ADX 48 (strong up-trend), 5-min ADX 39 (firm) . The aggregate 1-min technical signal has flipped to “neutral” only because momentum oscillators are stretched, not because trend has weakened.

*Volatility regime.* Daily ATR₁₄ sits at **6.40 USD** (≈ 3.8 %) . Expect roughly **±3.8 USD** of range into the NY close.

**2. Refreshed support / resistance**

| Level (USD)         | Origin & note                                 | Bias             |
| ------------------- | --------------------------------------------- | ---------------- |
| **176.80 – 176.50** | 0.6 × ATR extension above HOD                 | Fade first touch |
| **175.50**          | Intraday fib 1.618                            | Trim longs       |
| **174.24**          | VWAP-prox current print                       | Battleground     |
| **172.98**          | 1-min R cluster (was resistance, now support) | Buy on hold      |
| **170.20**          | Baseline day-close / weekly pivot             | Strong support   |
| **168.66**          | 1-min S cluster low                           | Major buy zone   |

(Levels pulled from `support_resistance.levels` in the AMD.json 1-min block and extended with ATR projections)&#x20;

**3. Intraday bias & setups (14:05 → 20:00 UTC)**
With price firmly above 172.98 the *primary bias is long*. Look to buy pull-backs into fresh support while being ready to fade a blow-off spike toward 176.8.

*Setup A — Pull-back long at 172.98*

* **Entry.** Limit-buy 173.00 ± 0.10 on a 1-min basing candle.
* **Stop.** 172.40 (0.5 × ATR₅ ≈ 0.6 USD).
* **Target-1.** 174.20 (re-test VWAP).
* **Target-2.** 175.50; trail remainder under 174.60.
* **Size.** (Acct × 1 %) / 0.60.

*Setup B — Momentum breakout*

* **Trigger.** 5-min close > 175.60 on rising volume.
* **Entry.** Market.
* **Stop.** 174.70 (low of the breakout bar).
* **Target-1.** 176.50.
* **Target-2.** 176.80 (ATR cap); exit all by 19:30 UTC.

*Setup C — Exhaustion fade*
If price tags **176.70–176.80** and prints a 1-min bearish engulfing:

* **Entry.** Limit-sell 176.75.
* **Stop.** 177.40 (0.65 USD).
* **Target.** 175.50 then 174.60; flat by session end.
* **Note.** Counter-trend; risk half-normal size.

**4. Pattern & momentum factors**

* An intraday *Triple Top* (bearish) matured earlier around 169.83 but price invalidated it by surging past the pattern stop-loss (170.25) — failure-to-fail strengthens the bull case .
* 15-min STOCH is > 80 (overbought) and RSI-5 min is diverging; watch for momentum cooling but respect the higher-time-frame trend.
* Daily MACD remains in a bullish zero-line cross, echoing the weekly up-swing bias from the baseline report .

**5. Timing grid (UTC)**

| Window          | Plan                                                                 | Objective                      |
| --------------- | -------------------------------------------------------------------- | ------------------------------ |
| **14:05-15:00** | Stalk pull-back into 173.0 (Setup A)                                 | Build core long                |
| **15:00-17:00** | If long, trail stops; add on breakout > 175.60 (Setup B)             | Ride trend to 176.5-176.8      |
| **17:00-18:30** | Monitor for climax candles; be cautious                              | Scale partials / tighten stops |
| **18:30-20:00** | Execute exhaustion fade if 176.8 tag (Setup C) or simply trail longs | Flat all by close              |

**6. Risk management checklist**

* Cap total open risk to **1 % E** across all positions.
* Reduce new size to 0.5 % E if VIX-adj 1-min range > 1 ATR₁₄ in < 30 min.
* Never add to losers; pyramid only after locking in ≥ 0.4 × ATR profit.
* Hard-flat all positions by 20:00 UTC — no overnight holds.

This framework keeps you aligned with the dominant intraday up-trend while guarding against a late-session blow-off.
