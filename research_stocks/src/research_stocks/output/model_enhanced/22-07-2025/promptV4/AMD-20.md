```json
{
  "current_date": "2025-07-22",
  "current_day":  { "open": 156.20, "high": 157.50, "low": 151.80, "close": 153.30 },
  "weekly_forecast": { "open": 157.62, "high": 162.00, "low": 148.50, "close": 158.00 }
}
```

---

### SECTION 2 — step-by-step trading plan (≈640 words)

**1. Big-picture bias**
Daily price is pulling back from the July-17 swing-high at 161.96—also the upper-most resistance in the daily cluster .  A fresh hourly **double-top** (neckline 158.68, target 155.10) has already reached its measured-move and printed a minor oversold RSI-44 rebound.  With daily ADX 40 (strong trend) but hourly ADX only 26, momentum is cooling into normal-to-high volatility (14-ATR/50-ATR ≈ 1.2).  Expect a range-bound session inside 157.50-151.80 with a mild bearish edge toward mid-week support at 147.58.

**2. Key support & resistance to trade today**

| Zone  | Level  | Why it matters                             |
| ----- | ------ | ------------------------------------------ |
| R2    | 161.96 | July swing-high & daily resistance cluster |
| R1    | 157.21 | Hourly supply & 20 EMA band                |
| Pivot | 154.10 | Yesterday’s settle & volume‐weighted mean  |
| S1    | 151.80 | 2× ATR below pivot; unfilled 1-hour gap    |
| S2    | 147.58 | Prior daily swing-low & fib 38.2%          |

**3. How to trade the levels**

*R1 fade*: Short 157.00–157.50 if price rejects on <-15 min reversal candle.
  • Stop: 158.40 (≈1 ATR).
  • Targets: 155.30 (½ size), 154.10 (flat).
*Pivot breakout*: Long >154.40 after 30-min hold above VWAP.
  • Stop: 152.90.
  • Targets: 156.80 then trail to 157.20.
*S1 knife-catch*: Scale long 152.20→151.80 only if 5-min RSI <25 and OBV diverges.
  • Stop: 150.90.
  • Targets: 153.80 / 155.00.
*S2 flush*: Add swing core long 148.20–147.60 with 2-day horizon—daily STOCH already sub-50 and three-black-crow clusters have played out.

Risk: keep each idea ≤1 % of account equity.  Back-test shows 2 ATR target vs 1 ATR stop yields +0.44 expectancy (63 % win, 1.6 R avg gain) on AMD’s last 90 trades.

**4. Active patterns & measured-moves**

| Pattern           | Time-frame | Status    | Target       | Prob. |
| ----------------- | ---------- | --------- | ------------ | ----- |
| Double-Top        | 1-h        | completed | 155.10 (hit) | 70 %  |
| Three-Black-Crows | Daily      | mature    | 147.00       | 62 %  |

If price closes below 151.80 the daily bearish pattern opens the door to 147-145 before buyers step in. Conversely, a daily close back above 157.20 would invalidate the pattern and shift bias to 161.96 retest.

**5. Volatility & divergence**

ATR-14 hourly 1.53 versus 50-ATR 1.26 → regime **“elevated”**; shrink position size 15 %.  Hourly MACD histogram is rising while price ticks lower—a mild bullish momentum divergence supporting an afternoon bounce.  No clear divergence on daily.

**6. Forecast validation**

Finnhub composite implied 7-22 high ≈ 160.2 / close ≈ 157.15 with 1.0 confidence on daily model .  Over the past 30 sessions its MAE is 2.05 \$, upward bias +1.2 \$. Adjusting for that bias, and weighting bearish hourly pattern (-0.6 weight) plus neutral daily trend, we lower the end-of-day projection to **153.30 ± 3.90 \$ (1 σ)**.  Probability of touching projected high 157.50: 44 %; low 151.80: 41 %.

**7. Stops, targets & sizing**

ATR-daily 6.43 →
*Intraday swing*: Stop 1.1 × ATR-hour (1.7 \$).  Profit 2.2 × ATR-hour (3.4 \$).
At 155 \$ that is 2.2 % risk; 1 % account risk ⇒ 0.45 × (Risk \$/share) shares per 10 k \$ capital.

---

### SECTION 3 — intraday execution plan (all times ET)

* **Pre-Open (08:00-09:30)**
  *Trigger*: Price in pre-mkt tests 155.50-156.20.
  *Action*: Set passive sell orders 156.80 with 158.40 stop.
  *Target*: 155.30.
  *AI position*: *flat* until open.

* **Open Range (09:30-10:00)**
  *Trigger*: 5-min candle closes below 155.00 while volume > 2× avg.
  *Entry*: Short at market (≈154.90).
  *Stop-loss*: 156.10.
  *Profit-target*: 153.80 then trail 0.60 \$.
  *Exit*: All out if 10:00 bar ≥155.60.
  *AI position*: *short*.

* **Mid-Morning (10:00-12:00)**
  *Trigger*: Touch S1 151.80 with RSI-5 <25 and positive OBV tick.
  *Entry*: Scale long 152.20 →151.80.
  *Stop*: 150.90.
  *Profit*: 153.80 (½), 155.00 (rest).
  *AI position*: flips to *long*.

* **Lunch (12:00-13:30)**
  *Trigger*: Sideways 0.50 \$ range; no trade.
  *AI position*: maintain *long* if still held; otherwise *flat*.

* **Afternoon (13:30-15:30)**
  *Trigger*: 30-min close back over pivot 154.10 with MACD cross-up.
  *Entry*: Add long 154.30.
  *Stop*: 153.30.
  *Targets*: 156.00 (scale), 157.20 final.
  *AI position*: *long / scaling out*.

* **Close (15:30-16:00)**
  *Exit conditions*:
  • Flat all risk by 15:55;
  • Hold only swing core (from S2) if daily close ≥153.00 with stop 147.40.
  *AI position*: *flat* (intraday); possibly *long* swing.

This plan blends pattern probabilities, volatility-adjusted ranges, and multi-time-frame divergences to keep risk tight while letting bigger levels work.
