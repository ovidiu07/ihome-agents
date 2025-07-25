```json
{
  "current_date": "2025-07-25",
  "current_day": {           // **re-forecasted with pre-open drift**
    "open": 164.66,          // unchanged – official floor print (08:00 ET)
    "high": 166.70,          // weighted projection: yesterday’s R-3 + 1-σ ATR
    "low": 162.30,           // confluence of pre-market floor & 15 min value-area low
    "close": 166.10          // bias-adjusted VWAP-to-VWAP continuation
  },
  "weekly_forecast": {       // fine-tuned using today’s pre-open momentum weight (α = 0.25)
    "open": 157.62,
    "high": 172.40,          // +0.8 % uplift vs prior on volatility regime shift
    "low": 154.90,           // -0.1 % tweak from stronger mid-week bid
    "close": 169.40
  }
}
```

---

## Step-by-step trading plan  (≤ 650 words) — **REFRESHED**

**Key intraday levels (pre-open scan 11:45 UTC)**

* **Micro (1-min)** – VWAP 163.74, S 162.35 / 163.18, **pivot 164.20**, R 164.66, major R 165.85, R2 167.18
* **Short-term (5-min)** – value-area low 162.83, control 164.20, V-A high 165.40; ceiling cluster 165.85-166.10
* **15-min / Hourly** – balance zone 161.80-164.20; imbalance trigger 166.00; yesterday’s high 166.50 = weekly R-2
* **Daily / Weekly** – composite supply 167.18-167.80; structural support 159.44-160.35 (un-tested gap)

| Zone (price)      | Bias / intention                 | Entry trigger (volume + price action)           | Initial SL | First TP | Runner TP |
| ----------------- | -------------------------------- | ----------------------------------------------- | ---------: | -------: | --------: |
| **162.30 ± 0.10** | Long scalp into VWAP             | Flush & 1-min bullish engulf; > 150 k vol       |     161.80 |   163.18 |    164.20 |
| **164.66-164.85** | Fade short (mean-reversion band) | Failed 1-min breakout / lower-high under 165.00 |     165.20 |   163.18 |    162.35 |
| **165.85-166.10** | Breakout long to R-2 ext         | 5-min close > 166.10 **and** OBV ↑ above VWAP   |     165.20 |   166.70 |    167.18 |

**Live patterns**

* **1-min Double-Bottom** (bullish) matured ➜ magnet 165.51, extension 166.31 pending.
* **1-min Triple-Top** at 165.85 (bearish) – invalidate with 5-min close > 166.10.
* **5-min Triangle** (bullish) – apex break targets 167.18.
* No new hourly formations – still digesting yesterday’s impulse.&#x20;

**Volatility & risk**

* **ATR-14** – 1-min ≈ 0.46 pt · 5-min ≈ 1.16 pt · 15-min ≈ 1.31 pt.
* Daily expected range 5.1 pt (1.05 × 20-day σ).
* **Stops** ≥ 0.5 pt (0.45 × 5-min ATR) to dodge noise.
* **ADX** – Hourly 19.6 (non-trending) ⇒ lean on range until 166.10/162.30 breaks.

**Divergences**

* Hourly MACD still < 0 while price flirts with 165+   → regular bearish divergence into 165.85 cap.
* 5-min RSI flattening near 60 as price creeps up – momentum waning inside 164.66-165.00 band.

**Position sizing**
Risk 1 % equity per idea. Example: 0.60 pt stop → shares = 0.01 × Acct Eq / 0.60. Minimum R\:R ≥ 2:1.

**Pivot & Fibonacci (re-computed)**

* **Daily pivot** = (166.50 + 160.90 + 165.20) / 3 ≈ 164.20.
* **Fib retrace** (162.83 → 165.40) – 38 % = 164.42 (≈ pivot), 61 % = 163.81.
* **Extensions** – 127 % = 166.70, 161 % = 167.80.

**Bull vs Bear roadmap**

* **Bull case (52 %)** – Hold > VWAP 163.74, break 165.85 → 166.70 / 167.80.
* **Bear case (48 %)** – Slip < 163.18, lose 162.83 → 162.30 / 161.80 sweep.

---

### Intraday execution plan — **optimised for 25 Jul 2025**

* **Pre-Open (< 14 :30 UTC)**   Mark VWAP 163.74 & 162.30 floor. *AI: flat*
* **Open Range (14 :30-15 :00)** Long sweep-&-reclaim 162.30; SL 161.80; TP 163.81 ➜ trail. *AI: long on trigger*
* **Mid-Morning (15 :00-16 :30)** Scale ½ at 164.20 pivot; watch OBV; exit rest if 164.66 stalls. *AI: partial-long / flat*
* **Lunch (16 :30-18 :00)** Fade 164.66-164.85 spike; SL 165.20; target 163.18. *AI: flat/short scalps*
* **Afternoon (18 :00-20 :00)** Momentum long only on 5-min close > 166.10 (+ 2× vol); pull-back bid ≥ 165.95 to 166.70/167.18. Else, range shorts. *AI: dynamic*
* **Close (20 :00-20 :30)** Flatten all; scalp only if price deviates ±0.50 pt off VWAP on burst vol. *AI: flat*

---

## SECTION 4 — Based on the further data, **here is how you should trade today**

| Epoch (UTC)     | Entry trigger & confirmation                                                   | Stop-loss (ATR / SR) | Profit-target & scaling                               | AI state             |
| --------------- | ------------------------------------------------------------------------------ | -------------------- | ----------------------------------------------------- | -------------------- |
| **Pre-Open**    | *Stand-by* – pre-market liquidity thin; build watch-list & alerts.             | —                    | —                                                     | **Flat**             |
| **Open Range**  | **Long** 162.30 flush + 1-min engulf + > 150 k vol **or** 5-min close > 163.00 | 161.80 (1× ATR-5)    | ½ off 163.81; trail to 164.20 pivot; runner to 164.66 | **Long**             |
| **Mid-Morning** | Add long on firm 5-min hold > 164.20 + OBV rising                              | 163.81               | Target 164.85; trail rest at 164.42 (38 % fib)        | **Long / scale-out** |
| **Lunch**       | **Short fade** 164.66-164.85 double-top (two 1-min LH + vol drop)              | 165.20               | Cover ½ 163.18; runner 162.30                         | **Short**            |
| **Afternoon**   | **Momentum long** 5-min close > 166.10 **and** 2× vol ⇒ bid pull-back ≥ 165.95 | 165.20               | First TP 166.70; runner 167.80 (161 % ext)            | **Long**             |
| **Close**       | Square positions; scalp only VWAP ± 0.50 pt bursts (≤ 0.80 pt objective)       | 0.40 pt              | Flat into bell                                        | **Flat**             |

**Risk template**: 1 % account risk; use 0.6 pt default stop to size.
**High-impact SR recap**: 162.30 / 163.18 support — 164.20 pivot — 164.66/165.00 supply — 165.85 cap — 167.18 weekly R-2.
**Active setups**: watch 5-min Triangle apex > 166.10 for squeeze; monitor 1-min Triple-Top for fade.

---

*Every element from the prior report has been retained and updated with the latest 1-, 5-, 15-minute data, recalibrated ATR regimes, merged support/resistance, refreshed pivot/Fibonacci figures, live VWAP and divergence checks, plus the new epoch-by-epoch Section 4 trading blueprint.*
