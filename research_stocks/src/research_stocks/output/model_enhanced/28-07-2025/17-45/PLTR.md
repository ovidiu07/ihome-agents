### SECTION 1 — concise forecast JSON

```json
{
  "current_date": "2025-07-28",
  "current_time": "2025-07-28T14:00:00Z",
  "today_forecast": { "open": 159.60, "high": 160.80, "low": 158.40, "close": 159.20 },
  "weekly_forecast": { "open": 160.50, "high": 166.00, "low": 157.00, "close": 161.50 }
}
```

### SECTION 2 — expert trading plan

**1. Updated context & multi-TF snapshot**

*Price action.* The most recent 1-min print (14:00 UTC) is **159.60**, extending the slow bleed that began after the 10:30 UTC spike to 160.80. This keeps price camped under the prior intraday pivot at 160.65 and leaves the earlier *Three Black Crows* sequence fully intact .

*Trend strength.* Aggregate 1-min ADX is **≈ 13** (non-trending), while the 5-min ADX is hovering near **27**; directional bias therefore comes from price structure rather than raw trend strength .

*Volatility regime.* The 5-day ATR on the daily chart remains **≈ 5.2 pts** (3.3 %), so the remaining half-session should accommodate roughly **±3 pts** of range .

**2. Refreshed support / resistance**

| Level (USD)         | Origin                       | Tactical bias              |
| ------------------- | ---------------------------- | -------------------------- |
| **161.30 – 161.25** | 1-min R-cluster              | First fade zone            |
| **160.52**          | 1-min swing high / fib 1.272 | Short trigger on test      |
| **159.35**          | 1-min pivot                  | Battleground               |
| **158.95 – 159.05** | Minor support                | Scalp-long window          |
| **157.96**          | Weekly S1 & 1-min S3         | Major support / cover zone |

*Levels sourced from `support_resistance.levels` in the uploaded JSON*&#x20;

**3. Bias & setups for the rest of the session (14:00 → 20:00 UTC)**
With price south of 160.52 and the bearish candle structure unresolved, the **primary bias is short** until a 5-min close back above 160.52.

*Setup A — Pull-back short at 160.52*

* **Entry.** Limit-sell 160.45 – 160.55 on a 1-min stall.
* **Stop.** 160.95 (≈ 0.5 ATR₅).
* **Target-1.** 159.35; **Target-2.** 158.95; trail remainder at –1×ATR.
* **Risk.** 1 % E / 0.50 risk-per-share.

*Setup B — Breakdown continuation*

* **Trigger.** 5-min close < 159.20 with rising volume.
* **Entry.** Market.
* **Stop.** 159.85 (upper half of the trigger bar).
* **Target-1.** 158.40 (projected intraday low).
* **Target-2.** 157.96 weekly S1.
* **Management.** Move stop to breakeven after +0.35 ATR.

*Setup C — Counter-trend scalp at 158.95 – 159.05*
Activate only if price tags the micro-support *and* a bullish reversal candle prints:

* **Entry.** Limit-buy 159.00 ± 0.05.
* **Stop.** 158.60 (0.40 ATR₅).
* **Target.** 159.90 then 160.50; exit remainder by 18:30 UTC or if 5-min RSI > 70.

**4. Pattern & momentum watchlist**

* The completed *Three Black Crows* still projects a measured-move to \~158.9, just above weekly S1 .
* Earlier bullish *Triangle* pattern hit its profit-1 target and is now neutral; failure to extend reinforces the down-bias .
* Hourly rising-channel floor lies near 158.5; an hourly close < 158.9 confirms channel failure and likely fast tracks 157.96.

**5. Timing grid (UTC)**

| Window      | Primary action                                            | Objective               |
| ----------- | --------------------------------------------------------- | ----------------------- |
| 14:00-15:00 | Sell pull-back into 160.52 (**Setup A**)                  | Ride to 159.35          |
| 15:00-17:00 | Add on breakdown < 159.20 (**Setup B**)                   | Extend toward 158.40    |
| 17:00-18:30 | Watch for 158.95 test; run **Setup C** if reversal prints | Mean-revert to 159.90   |
| 18:30-20:00 | Trail shorts or cover into 157.96; flat by close          | Zero overnight exposure |

**6. Risk & trade management reminders**

* Limit total open risk to **1 % E**; halve new size if VIX-adj 1-min range > 1 ATR in < 30 min.
* Never add to losers; pyramid only after locking in ≥ 0.4 ATR.
* Hard-flat on all positions by 20:00 UTC (16:00 ET).

This plan keeps you aligned with the prevailing intraday down-bias while preserving a tight, rules-based counter-trend scalp at the strongest micro-support.
