### SECTION 1 — concise forecast JSON

```json
{
  "current_date": "2025-07-28",
  "current_time": "2025-07-28T14:00:00Z",
  "today_forecast": { "open": 159.60, "high": 160.80, "low": 158.40, "close": 159.30 },
  "weekly_forecast": { "open": 160.50, "high": 166.00, "low": 157.00, "close": 162.00 }
}
```

### SECTION 2 — expert trading plan

**1. Updated context & multi-TF read**

*Price action.*  The last 1-min close at 14:00 UTC printed **159.60** after a fast slide from the 160.80–161.00 zone earlier in the session .  That break has cracked the prior intraday pivot at 160.65 and completed a *Three Black Crows* sequence, a reliably bearish continuation pattern .

*Trend strength.*  Intraday trend metrics remain firm (ADX-1 min ≈ 28, ADX-5 min ≈ 39, both “trending”) .  Momentum, however, has flipped negative on the most recent MACD histograms (5-min) and OBV tick-down.

*Volatility regime.*  The 5-day ATR on the daily chart sits at **5.22 pts** (≈3.3 %) , so a *half-day* intraday range of roughly 0.6 × ATR → **3 pts** is expected into the NY close.

**2. Refreshed support / resistance**

| Level (USD)       | Source & note                     | Bias                   |
| ----------------- | --------------------------------- | ---------------------- |
| **161.30-161.25** | 1-min R cluster, prior session R1 | Sell into first test   |
| **160.76**        | 1-min swing high, fib 1.272       | Initial short trigger  |
| **159.60**        | VWAP-prox last print              | Tactical battleground  |
| **159.04**        | 1-min S cluster                   | Buy scalp / stop mover |
| **157.96**        | Weekly S1 & 1-min S3              | Major swing support    |

(Levels from the real-time S/R array in the JSON 1-min block )

**3. Bias & setups for the rest of the session (14:00 → 20:00 UTC)**
With price **below 160.76** and bearing fresh bearish pattern confirmation, the *primary bias is short* until a 5-min close re-establishes above that level.

*Setup A — Reversion short at 160.76*

* **Entry.** Limit-sell 160.70-160.80 on a 1-min stalling candle.
* **Stop.** 161.10 (≈0.5 ATR).
* **Target-1.** 159.60; **Target-2.** 159.10; trail runner at –1×ATR.
* **Position size.** (Acct × 1 %) / 0.50 ≈ *Risk units*.

*Setup B — Breakdown continuation*

* **Trigger.** 5-min close < 159.30 with rising volume.
* **Entry.** Market.
* **Stop.** 160.05 (upper half of prior 5-min bar).
* **Target-1.** 158.40 (projected intraday low).
* **Target-2.** 157.96 weekly S1.
* **Management.** Move stop to breakeven once +0.4 ATR in profit.

*Setup C — Counter-trend scalp at 159.05-158.95*
Activation only if price tags the 159.04 support cluster *and* a bullish reversal pattern prints on 1-min:

* **Entry.** Limit-buy 159.05 ±0.05.
* **Stop.** 158.70 (0.35 ATR).
* **Target.** 159.90 then 160.70; flatten remainder by 18:30 UTC or if 5-min RSI crosses 70.

**4. Pattern & momentum watchlist**

* Three Black Crows on 1-min is already mature; respect its measured-move (≈ –1.0 pt) which aligns with the 159.04 level.
* Failed bullish *Double Bottoms* earlier today emphasize that upside follow-through is weak .
* Hourly structure remains in a rising channel that bottoms near 158.5; any hourly close < 158.9 would constitute a channel break and open 157.9 quickly.

**5. Intraday timing grid (UTC)**

| Window      | Plan                                                        | Objective                   |
| ----------- | ----------------------------------------------------------- | --------------------------- |
| 14:00-15:00 | Look for pullback into 160.70-160.80 and run **Setup A**    | Capture first leg to 159.60 |
| 15:00-17:00 | Momentum phase; add **Setup B** on break < 159.30           | Press toward 158.40         |
| 17:00-18:30 | Lunch drift; if price reaches 159.05-158.95 set **Setup C** | Mean-revert to 159.90       |
| 18:30-20:00 | Trail shorts; exit all ahead of close                       | Flat EOD                    |

**6. Risk & trade management reminders**

* 1 % equity risk per idea.
* Never add to a losing position; pyramid only after locking in ≥0.4 ATR.
* If VIX-adjusted tick range exceeds 1 ATR in <30 min, halve size on new entries.
* ALL positions flat by the 20:00 UTC bell.

This plan keeps you aligned with the prevailing intraday down-bias while retaining a concise counter-trend play at the strongest support cluster.
