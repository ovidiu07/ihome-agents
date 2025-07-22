```json
{
  "daily_forecast":  { "open": 154.01, "high": 159.50, "low": 152.50, "close": 158.00 },
  "weekly_forecast": { "open": 153.71, "high": 160.00, "low": 149.00, "close": 156.00 }
}
```

**Volume & Momentum snapshot** – Intraday OBV turned up after yesterday’s liquidation but is still below last week’s peak, flagging only moderate accumulation . Hourly ADX sits near 29, confirming a maturing but still viable trend . One-minute VWAP (154.3 ≈ live) now aligns with the daily pivot, so price acceptance above that line is required for longs.

**Key support / resistance**

| Level           | Why it matters                                    | How to trade it                                         |
| --------------- | ------------------------------------------------- | ------------------------------------------------------- |
| 149.33–151.09   | One-minute double-bottom base + minor SR cluster  | Buy‐reaction zone; 1 × ATR(1 m)=0.31 stop below 149.18  |
| 152.5           | Enhanced low & daily S1                           | Scale-in long if momentum turns up                      |
| 153.19          | Daily pivot (P)                                   | Intraday bias filter – above = long                     |
| 154.80–155.10   | Yesterday’s midpoint + open-range high            | First upside scalp target                               |
| 157.21          | Hourly swing pivot                                | Cover shorts / fade longs on first test                 |
| 161.96 / 162.26 | Daily & hourly ceiling                            | Only engage on confirmed breakout with volume > 2 × avg |

**Patterns & scenarios**
*Hourly* double-top completed its 155.10 profit target and is now neutral . *Daily* bearish candle groups (three-black-crows) are aging, while a fresh one-minute *double-bottom* succeeded this morning – a micro bullish tail-wind . Measured-move math projects 158.9 on the upside and 149.3 on the downside; those align with today’s forecast bounds.

**Fibonacci & pivots**
Last swing high/low (156.23 ↔ 149.34) plots 38 % retrace at 153.80 and 61 % at 151.77 – both sit inside the support ladder and strengthen the buy-the-dip case. Camarilla H3/L3 are 156.46 / 152.03; use them for tight day-trades.

**Volatility & risk**
Daily ATR(5) ≈ 5.38 (3.5 % of price) → high-vol regime; hourly ATR(14) 1.73 → normal; one-minute ATR 0.31 → low – moderate . Position-size off daily ATR: risking 1 % means \~ 0.5 × ATR ≈ \$2.70 stop for swing trades ⇒ \~ 37 bps of capital per share.

**Bull vs. Bear likelihood**

| Case         | Target      | Prob. |
| ------------ | ----------- | ----- |
| Bull-stretch | 159.5 / 160 | 45 %  |
| Range-hold   | 154–158     | 35 %  |
| Bear-flush   | 151         | 20 %  |

**Forecast validation** – Finnhub’s raw 30-day MAE clocks 2.05 \$; bias +0.12 \$ up-side. Our ATR-weighted blend therefore nudges today’s close 0.9 \$ above their 157.07 call to 158.0 . One-sigma band: 152.6–159.4 (prob. high/low hit ≈ 32 %).

---

### Intraday execution plan (all times ET)

* **Pre-Open ( < 09:30 )**
  *Trigger*: No trade – mark VWAP & overnight high/low.
  *AI position*: flat.

* **Open Range 09:30-10:00**
  *Entry*: Long on 1-min close > VWAP + volume-spike (≥ 2 × 5-min avg).
  *Stop*: 1-min ATR14 (0.31) below entry or sub-149.18.
  *Targets*: 154.80 first, 155.80 second (½ scale).
  *Exit*: If price falls back under VWAP.
  *AI position*: long if triggered; else flat.

* **Mid-Morning 10:00-12:00**
  *Entry*: Fade into 157.21 hourly pivot if RSI(5 m) > 70.
  *Stop*: 0.7 × hourly ATR ≈ 1.2.
  *Target*: 155.60, trail remainder.
  *AI position*: short-scalp; flat after cover.

* **Lunch 12:00-13:30**
  *No new trades* – volatility trough; manage open runners.
  *AI position*: flat.

* **Afternoon 13:30-15:30**
  *Entry*: Breakout buy > 157.21 with 5-min close & ADX(5 m) > 25.
  *Stop*: 155.90 (last minor HL).
  *Targets*: 158.90 (pattern measured move), 159.50.
  *AI position*: long on breakout confirmation.

* **Close 15:30-16:00**
  *Exit conditions*: RSI(5 m) > 70 or price < VWAP – flat all.
  *AI position*: flat by session end.
