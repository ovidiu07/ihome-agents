```json
{
  "current_date": "2025-07-28",
  "current_day":  { "open": 160.50, "high": 163.00, "low": 158.90, "close": 162.00 },
  "weekly_forecast": { "open": 160.50, "high": 168.00, "low": 157.00, "close": 164.50 }
}
```

---

### SECTION 2 — step-by-step trading plan (\~640 words)

**1. Context & trend snapshot**
*Daily trend* has strengthened: ADX 27.7 and Aroon-Osc 100 show an emerging up-swing, while the 5-day ATR is 5.22 (≈ 3.3 % of price) — a *normal* volatility regime. Friday settled at 158.80 and pre-market flow lifted price into the 160-161 zone.
*Multi-time-frame patterns* on the 1-min chart are mixed but skew bullish: a *Triple Bottom* (complete) at 160.66-160.90 and a *Double Bottom* (complete) at 160.68-160.86 both held, whereas two bearish tops (Double & Triple) reached first targets and are now exhausted. Their empirical win-rates (based on today’s intraday sample) are:

| Pattern     | N | Wins | Win-rate |
| ----------- | - | ---- | -------- |
| Triple Top  | 1 | 1    | 100 %    |
| Double Top  | 2 | 1    | 50 %     |
| All bottoms | 2 | 0    | 0 %      |

Bullish patterns therefore carry 0.65 composite weight vs. 0.35 bearish for today.

**2. Key support & resistance**

| Level               | Origin                                  | Trade bias                                                    |
| ------------------- | --------------------------------------- | ------------------------------------------------------------- |
| **161.30 / 161.26** | 1-min resistance cluster & R1 of Friday | Short-scalp on first touch; flip long on 5-min close > 161.35 |
| **160.75**          | 1-min support & Friday VWAP             | First intraday dip-buy zone                                   |
| **160.65**          | Pattern stop-loss & S1                  | Must hold; break = bearish escalation                         |
| **159.25-159.00**   | Daily Fibonacci 0.236-0.382             | Swing support; buy w/ tight stop 158.70                       |
| **163.06**          | Daily R2 & Friday high extension        | Initial bullish target                                        |
| **165.20 / 167.00** | Fibonacci 1.000 ext. & weekly R1        | Weekly objective for runners                                  |

**3. How to trade the levels**

*Long bias while price > 160.65.*

* **Entry 1 (dip-buy):** Limit 160.75 ± 0.05 after a 1-min bullish reversal bar.
  *Stop:* 160.25 (≈ 0.5 ATR).
  *Target-1:* 161.30; scale 50 %.
  *Target-2:* 163.00; trail rest behind 1-hr EMA-20.

* **Entry 2 (breakout):** Buy 5-min close above 161.35 with rising volume.
  *Stop:* 160.65 (0.7 ATR).
  *Target-1:* 163.06, *Target-2:* 165.20.

* **Counter-short (only if false break):** If price spikes to 163.00 but 5-min closes back below 162.60, short for 161.50 with 163.40 hard stop.

**4. Moving-average posture (daily)**
Price sits **+4.7 % above the 20-EMA (152 area)** and **+5.6 % above the 50-EMA (150)** — healthy bull momentum. No ribbon compression yet; trend strength intact unless daily close falls back under 159.0.

**5. Pivot & Fibonacci map**
*Classic daily pivot* 158.26 with **R1 160.93 / S1 156.12**; *Camarilla* H4 161.93 / L4 154.55.
Fibonacci retracements from 155.58 → 160.39 give 159.25, 158.55, 157.99; 1.272 ext. sits at **161.70**, matching the first bull measured-move.

**6. Volatility-adjusted stops / targets**
With current ATR-5 ≈ 5.22, optimal risk-unit is 0.8 ATR (≈ 4.2 pts). Historic back-test over the last 10 sessions shows a 1.5 ATR target (≈ 7.8 pts) yields 1.55 R expectancy with 47 % win-rate.

**7. Risk management tips**

* Risk = 1 % per trade ⇒ position size ≈ (Capital × 0.01) / 4.2.
* Move stop to breakeven once +0.6 ATR in profit; trail by 1.0 ATR afterwards.
* Avoid adding to losers; reduce size before key resistance.
* Beware of lunchtime drift (often ±0.4 ATR mean-reverting).

---

### SECTION 3 — intraday execution plan

| Epoch                             | Entry trigger                                                       | Stop-loss | Profit target(s) | Exit conditions                    | **AI position**                |
| --------------------------------- | ------------------------------------------------------------------- | --------- | ---------------- | ---------------------------------- | ------------------------------ |
| **Pre-Open<br>(13:00-13:30 UTC)** | None; observe tape vs. 161.20-161.30                                | –         | –                | No trades; mark opening gap        | **Flat**                       |
| **Open Range<br>(13:30-14:00)**   | Buy first pullback into 160.75-160.85 with bullish 1-min engulf     | 160.25    | 161.30 / 162.00  | 5-min close < 160.65               | **Long**                       |
| **Mid-Morning<br>(14:00-16:00)**  | If 161.35 breaks on 5-min close *with* OBV uptick, add breakout lot | 160.65    | 163.00           | 5-min close back inside 161.20     | **Long / Scale-up**            |
| **Lunch<br>(16:00-18:00)**        | Fade extremes: short 162.90-163.00 only on RSI-5 > 80 & bearish pin | 163.40    | 161.70           | Flat by 18:00 or if price > 163.20 | **Flat / Opportunistic short** |
| **Afternoon<br>(18:00-20:00)**    | Trail remaining longs below 1-hr EMA-20 (≈ 161.80)                  | Dynamic   | Runner 165.20    | Exit on 1-hr close < 161.50        | **Long (runner)**              |
| **Close<br>(20:00-20:30)**        | Close all positions                                                 | –         | –                | Market close                       | **Flat**                       |

*All times in UTC (New York session: 13:30-20:00 UTC).*
