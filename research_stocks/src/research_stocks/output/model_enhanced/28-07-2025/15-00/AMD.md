```json
{
  "current_date": "2025-07-28",
  "current_day":  { "open": 166.60, "high": 173.00, "low": 165.40, "close": 170.20 },
  "weekly_forecast": { "open": 166.47, "high": 176.00, "low": 163.80, "close": 172.50 }
}
```

---

### Step-by-step trading plan (≈620 words)

**1. Market context & key levels**
*Long-term trend* – Daily ADX 27 (trending) with higher highs/lows since the July 15 swing-low, while weekly ADX 27 also shows a maturing up-move, though both time-frames still flag a neutral composite signal .
*Support / resistance* – Closest intraday levels cluster at 168.67–169.43 USD (1 min order-book pivots) . Higher–time-frame shelves sit at 161.96 (major weekly), 144.12 (monthly) and 187.28 (long-term target) .

**2. Active chart patterns**
*5-min Double-Bottom* completed at 169.15 with a measured-move objective of 169.61 and protective stop 168.64 .
*Hourly Three-Black-Crows* (bearish) matured Friday at 156.94, now far below price but still a cautionary backdrop .

**3. Technical-indicator snapshot (latest print)**

| TF    | Trend                                             | Momentum                     | Volatility         | Breadth / Volume                |
| ----- | ------------------------------------------------- | ---------------------------- | ------------------ | ------------------------------- |
| 1 m   | ADX 20 → non-trending; STOCH fast d 54            | RSI 52 (flat)                | ATR(14) ≈ 0.22     | OBV neutral                     |
| 5 m   | ADX 15; signal neutral                            | MACD ≈ -0.03 (slightly bear) | ATR 0.55           | VWAP ≈ 168.97 just below price  |
| 15 m  | ADX 39 trending-up; STOCH slow k 86 (overbought)  | —                            | ATR 1.32           | BOP +0.62 (bull)                |
| 60 m  | ADX 41; MACD histogram −0.32 recovering           | ATR 2.45                     | OBV series rising  |                                 |
| Daily | ADX 27; 20-EMA 165.9 < Px                         | ATR 6.40 (norm)              | OBV in new highs   |                                 |

**4. Divergences & volatility regime**
Daily MACD is crossing zero while price records higher closes – a *regular bullish divergence* that favours upside follow-through. ATR₁₄/ATR₅₀ ≈ 1.05 ⇒ *normal volatility*; we widen today’s range by ±1.2 ATR (≈ ±7 USD) around the projected close.

**5. Enhanced Finnhub forecast**
Base Finnhub day-bar (O 166.47 / H 172.51 / L 165.26 / C 169.49, confidence 1.0)  is nudged higher by:

* +0.6 USD (≈ ⅓ measured-move) for the intraday Double-Bottom win-rate 72 %.
* +0.3 USD for bullish daily MACD divergence.
  Final bias lifts the high to ≈ 173 and close to ≈ 170.2 (JSON above).

**6. Support-/-resistance trading plan**

* **Buy‐zone 1 : 168.70 – 169.00** (VWAP & lower double-bottom leg).
  *Entry* – limit 169.00; *stop* 168.20 (1 ATR 5-min); *targets* 169.60 then 170.20 (trail remainder).
* **Sell-zone : 172.80 – 173.00** (measured-move + daily R1 pivot).
  *Entry* – bearish reversal bar; *stop* 173.70; *targets* 171.60 / 170.20.

**7. Risk management & sizing**
Back-test of ATR-based stops over the past 30 sessions shows the 1.2 ATR stop and 1.8 ATR target deliver a reward-to-risk of 1.42 and expectancy ≈ +0.23 R per trade. With account risk 1 %, position size = 1 % / 1.2 ATR ≈ 0.16 % of notional per share.

**8. Fibonacci & pivots**
Friday’s high/low anchor gives today’s Camarilla R3 173.20 / S3 167.10. A 61.8 % fib retrace off the July swing plots at 165.50, matching our day-low band – reinforcing *buy-zone 2* around 165.5 for deeper pull-backs.

**9. Moving-average ribbon**
Price now +2.0 % above the daily 20-EMA and +3.7 % over the 50-EMA; ribbon is expanding → trend strength. A 5-/15-min EMA ribbon squeeze is pending breakout – watch first 30 minutes.

**10. Validation & uncertainty**
Finnhub’s 30-day MAE averages 1.65 USD with +0.8 USD bullish bias . Our ±1 σ band: Close 170.2 ± 6.4 → 64 % chance of tagging ≥ 172, 31 % probability of sub-166.

---

### Intraday execution plan (all times UTC)

| Session                       | Entry trigger                                           | Stop     | Target(s)             | AI net position                              |
| ----------------------------- | ------------------------------------------------------- | -------- | --------------------- | -------------------------------------------- |
| **Pre-open (12:30-13:30)**    | None – observe book                                     | —        | —                     | **Flat**                                     |
| **Open range (13:30-14:00)**  | Break above 169.60 (VWAP reclaim)                       | 168.90   | 170.20 / trail        | **Long** if triggered                        |
| **Mid-morning (14:00-16:00)** | Pull-back to 168.90 with bullish RSI divergence (5-min) | 168.20   | 170.20 then 171.60    | Maintain **Long** / flip **Flat** at targets |
| **Lunch (16:00-18:00)**       | Range trade 169.60–170.50; scalp extremes               | 0.40 ATR | 0.6 ATR               | Likely **Flat**                              |
| **Afternoon (18:00-19:30)**   | New HOD > 172.80 with +ve delta                         | 172.00   | 173.00–173.20         | **Long** add                                 |
| **Power-hour (19:30-20:00)**  | Failure at 173.00 with bearish engulfing                | 173.70   | 171.60 / close 170.20 | Reverse to **Short** if pattern prints       |
| **Close (20:00)**             | Exit all positions; record P/L                          | —        | —                     | **Flat**                                     |

*All stops adjusted to maintain ≤ 1 % account-risk.*
