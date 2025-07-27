```json
{
  "current_date": "2025-07-24",
  "current_day":  { "open": 153.8, "high": 154.3, "low": 153.3, "close": 154.1 },
  "weekly_forecast": { "open": 150.5, "high": 156.0, "low": 149.0, "close": 154.5 }
}
```

---

### SECTION 2 — step-by-step trading plan (\~620 words)

**Key support & resistance**

| Level  | Source                   | Notes                                                              |
| ------ | ------------------------ | ------------------------------------------------------------------ |
| 154.95 | 15-min SR ceiling        | Multi-TF confluence (also 5-min R4) – “line in the sand” for bulls |
| 154.10 | 1-min SR pivot           | Intraday trigger: prior evening-star apex                          |
| 153.83 | 1-min SR mid-band        | VWAP cluster; yesterday’s close                                    |
| 153.45 | 1-min SR floor           | First buy-the-dip zone (≈½-day ATR)                                |
| 151.78 | 15-min SR key support    | Triangle pattern base; weekly pivot                                |
| 149.47 | 15-min SR major support  | Fib 50 % of March–July leg                                         |

---

**How to trade the levels**

* **Long-bias scalps 153.45 – 153.60**
  *Entry*: Limit buy inside the band after a bullish 1-min reversal candle with RSI(14) < 30.
  *Stop*: 153.05 (≈0.40 — 13-period ATR on 1-min).
  *Target*: 154.00 first, trailer to 154.10.
  *R\:R*: \~1 : 1.5, probability ≈58 % thanks to high 15-min triangle win-rate.

* **Breakout continuation above 154.10**
  Evening-star on 1-min completed but failed to follow-through; a clean 5-min close > 154.10 invalidates the micro bearish pattern.
  *Entry*: Buy the first 1-min pull-back ≥ 154.10 with volume > 3-bar mean.
  *Stop*: 153.80 (just under VWAP).
  *Targets*: 154.50 (measured-move of failed double-top) then 154.95.
  If price reaches 154.50 quickly, scale half off and trail 20-period EMA on 5-min.

* **Fading extremes near 154.95**
  Triple-top on 5-min is mature and carries a 63 % historical win-rate. Look to short into 154.80–154.95 only after MACD(12-26-9) crosses down on 1-min.
  *Stop*: 155.15 (pattern stop-loss ﻿).
  *Targets*: 154.30, then 153.85 where bullish dip-buyers are expected.

---

**Active chart patterns & measured-moves**

* **15-min Symmetrical Triangle (bullish, complete)** projects +4.9 % → 158.08; pattern has already broken up and is retesting the apex, lending support to the intraday long bias.&#x20;
* **5-min Triple/Double Tops (bearish, complete)** give -1 – -1.6 % projections (152.6 & 152.5). They remain valid while price is below 154.95.&#x20;
* **1-min Evening-Star (bearish, complete)** lost downside momentum; failed patterns often fuel squeezes higher.

---

**Indicator & volatility snapshot**

* ADX < 20 on all three intraday frames ⇒ trendless chop; rely on support/resistance edges.&#x20;
* 15-min SMA-EMA ribbon is flattening; distance of price to 200 EMA ≈ -0.3 % – neutral.
* ATR(14) / ATR(50) ratio = 0.86 (low-normal regime).
   → Expand high/low projections by only 60 % of full ATR to avoid over-sizing ranges.
* RSI(14) divergent bullish on 15-min vs. price (higher RSI, lower price) – biasing forecasts slightly upward.
* Historical Finnhub MAE (past 30 days) = 0.42; bias +0.12 up-side – incorporated in forecast.

---

**Pivot, Fibonacci & risk**

* Daily classic pivots: S1 = 153.45, R1 = 154.10 – perfectly aligned with micro levels above.
* Fibonacci retrace of June swing 150.55 → 154.95: 38 % = 153.1; if 153.45 breaks, next magnet is 153.10.
* Trade sizing: with stop \~0.40 and 1 % account risk, size = (Equity × 0.01)/(0.40 × ATR-adj) ≈ 25 % of standard lot.

---

### SECTION 3 — intraday execution plan

* **Pre-Open (🕔 < 09 : 30 ET)**
  • Mark levels 153.45 & 154.10 on chart.
  • No orders; watch tape for opening imbalance.
  • *AI position: flat*

* **Open Range (09 : 30 – 10 : 00)**
  • If flush into 153.45 ± 0.05 with capitulation volume, open long per dip-plan.
  • Else, wait for 5-min ORB direction.
  • *AI position: long (only if dip triggers), otherwise flat*

* **Mid-Morning (10 : 00 – 12 : 00)**
  • Hold longs to 154.00-154.10; trail stop to breakeven once 153.85 prints.
  • If 154.10 breaks on a closing basis, add ½ size on PB to 154.00.
  • *AI position: long or flat depending on 154.10 break*

* **Lunch (12 : 00 – 14 : 00)**
  • Reduce position size by 30 % to avoid low-liquidity whips.
  • Keep stop 153.80, target 154.50.
  • *AI position: scaled-down long / flat*

* **Afternoon (14 : 00 – 15 : 30)**
  • On first test of 154.80-154.95: initiate tactical short (≤ ½ risk unit) with stop 155.15.
  • Manage legacy longs: exit remainder at 154.80 or move stop 154.30.
  • *AI position: hedge-short or flat*

* **Close (15 : 30 – 16 : 00)**
  • Cover any shorts by 15 : 55 regardless of target; fade into cash.
  • Flatten all positions by the closing auction.
  • *AI position: flat*

---

**End-of-day check-list**
If price settles > 154.50 the bullish weekly target (156.00) becomes base-case; below 153.30 keeps 152.50 bear scenario alive.
