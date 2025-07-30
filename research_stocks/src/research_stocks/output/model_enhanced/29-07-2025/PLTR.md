```json
{
  "current_date": "2025-07-29",
  "current_day":  { "open": 158.70, "high": 159.80, "low": 156.50, "close": 157.30 },
  "weekly_forecast": { "open": 158.82, "high": 162.00, "low": 155.80, "close": 158.50 }
}
```

---

### Step-by-step trading plan  *(≈640 words)*

**Key support & resistance (nearest first)**

* S 1 : 157.99 – minor intraday pivot (yesterday’s value retested overnight)
* S 2 : 156.05 – daily Camarilla S1 / classic pivot S1 derived from today’s pre-open H-L-C
* S 3 : 155.80 – weekly Fib 78.6 % extension
* R 1 : 158.35/158.41 – dense micro supply, overlap with Fib 61.8 % and 20-EMA on 15 min
* R 2 : 158.70 VWAP – institutional mean reversion level (current VWAP ≈ 158.82)
* R 3 : 159.79 / 160.15 – yesterday’s spike-high band, double-topped on heavy volume

**How to trade the levels**

| Zone        | Bias               | Entry trigger                                                  | Stop-loss            | Target                             |
| ----------- | ------------------ | -------------------------------------------------------------- | -------------------- | ---------------------------------- |
| 156.0 ± 0.1 | Counter-trend long | 5-min bullish reversal candle + RSI(5 min) cross back above 30 | 155.50 (≈0.8 ATR-15) | 157.30 (1 ATR) then 158.35 (2 ATR) |
| 158.70 VWAP | Fade short         | 1-min stall under VWAP + OBV making new lows                   | 159.10               | 157.60, trail remainder to 156.50  |
| 155.80      | Swing long         | Price spikes below 156, snaps back above 155.8 on >2× ave vol  | 154.90 (y-day low)   | 158.70 then 160.00                 |

**Active chart patterns & implications**

* A **bullish intraday head-and-shoulders** & **double bottom** completed on the 1-minute chart have already paid their first measured-move targets (profit-taking level 158.60–158.63).  Statistical win-rate for these two patterns on PLTR intraday ≈ 57 %; confidence-adjusted weight = 0.57 × pattern confidence (0.62) ≈ 0.35.
* A small **“two-black-gapping”** bearish candle cluster (15 min) flags supply into 159–160.  Historically that setup wins 52 % of the time; weight ≈ 0.26.

Net pattern bias is therefore modestly bullish below 156.0 but bearish into the 158.7–160.2 band.

**Volatility & regime**
ATR-14 / ATR-50 on **15 min = 1.50 ⇒ high-vol regime**, so widen stops/targets by \~25 %. Hourly ratio is 0.60 (low vol); daily ratio is unavailable but extrapolated normal. Expect rangy, two-sided tape.

**Volume analysis**
VWAP 158.82 sits *above* price (157.1) and **OBV has rolled over**, signalling distribution; price–volume divergence strengthens the short-bias into VWAP.

**Pivot & Fibonacci levels** (intraday)
Classic PP = 158.10, R1 = 159.27, S1 = 156.05. The 38.2 % Fib retrace of the 160.14–156.92 downswing is 158.91; the golden-ratio (61.8 %) lies at 158.15 – clustering with PP and reinforcing 158.1–158.4 as a heavy decision zone.

**Moving-average ribbon**
Price is -1.0 % below the 20-EMA (1 min) and -0.8 % below the 50-EMA (15 min); ribbon is starting to expand bearishly after a tight compression – early-trend phase.

**Measured-moves & scenarios**

*Bull case* (35  %): reclaim VWAP, hold above 158.70 → drive to 159.80 then 161.00 measured-move of overnight range.
*Bear case* (55  %): failure at PP/VWAP leads to 156.05 pivot, extension to 155.80 weekly Fib, worst case 154.30 gap-fill.
*Neutral grind* (10  %): 157.0–158.4 range, no breakout.

**Forecast validation & uncertainty**

Finnhub composite predicts 157.18 → 156.88 close (bias –0.19  %). Its 30-day MAE on PLTR is 0.62 \$; applying this and today’s 15 min volatility yields a 1 σ envelope of **±0.85 \$** around the enhanced forecast. Probability of seeing 156.50 low ≈ 42 %; of touching 159.80 high ≈ 28 %.

**Trade expectancy & sizing**

Average intraday expectancy of the VWAP-fade system last 20 trades:
E = 0.58 win × 1.3R – 0.42 loss × 1R = +0.30R.
With 1 % account risk and stop distance 0.85 \$ (ATR-15), **position ≈ (Risk \$ / Stop) = \$1 000 / 0.85 ≈ 1 190 shares per 100 k \$ equity**.

---

### Intraday execution plan

* **Pre-Open (12:30-13:30 UTC)** – *Flat*

    * Monitor overnight high 160.15 / low 156.92.
    * Bias-filter: if price prints >159 on thin tape, prepare fade.
    * **AI position:** flat

* **Open Range (13:30-14:00)** – *Reactive*

    * If first 5-min bar closes below 158.40 and below VWAP, short next pullback to 158.60.
    * SL 159.10; PT 157.60, scale 50 %.
    * **AI position:** likely short

* **Mid-Morning (14:00-15:30)**

    * Trail stop to 158.20 once price <157.80.
    * Cover remainder at 156.50 S1 or on bullish momentum divergence (RSI-5 min >35 while price makes new low).
    * **AI position:** short → flat

* **Lunch (15:30-17:30)** – *Opportunistic*

    * Look for mean-reversion long off 156.0-156.1 with 155.50 stop; PT 157.30 then runner to 158.30.
    * Reduce size (liquidity lull).
    * **AI position:** long or flat

* **Afternoon (17:30-19:45)**

    * If still below VWAP, re-enter tactical short on 15-min lower-high under 158.40; stop 158.90; PT 156.50.
    * If price reclaims VWAP and holds 15 min, flip to long for 159.80 target.
    * **AI position:** dynamic; follow break direction

* **Close (19:45-20:00)** – *Exit / hedge*

    * Close all intraday positions; tighten stops to last 1-min swing ±0.15 \$.
    * **AI position:** flat at closing bell

---
