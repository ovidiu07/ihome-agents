```json
{
  "current_date": "2025-07-24",
  "current_day":  { "open": 158.7, "high": 161.5, "low": 157.8, "close": 160.4 },
  "weekly_forecast": { "open": 158.65, "high": 164.0, "low": 154.8, "close": 160.8 }
}
```

---

### SECTION 2 — step-by-step trading plan (\~620 words)

**Key intraday support & resistance**

| Price             | Source                                                              | Rationale                                           |
| ----------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| **161.21–161.30** | 5-min & 15-min SR ceiling and prior double-top stop-loss cluster    | Major upside inflection; breakout triggers squeeze. |
| **160.82**        | 1-min SR upper band                                                 | VWAP-adjacent micro ceiling.                        |
| **160.18**        | 5-min SR pivot & ATR midpoint                                       | First pull-back buy zone inside range.              |
| **159.56**        | 1-min SR mid-band                                                   | Double-bottom neckline; strong magnet.              |
| **159.09**        | 1-min SR floor & bullish double-bottom “A” point                    | Must-hold for bulls.                                |
| **158.80**        | 5-min SR lower band                                                 | Half-day ATR from ORH; dip scalp area.              |
| **157.60**        | Weekly S1 derived from finnhub low 157.94 minus volatility haircut  | Last-defence support; below turns bias bearish.     |

---

**Active chart patterns & implications**

* **1-min Double Bottom** (incomplete, profit target ≈ 160.67) suggests near-term upside; stop-loss 159.01 .
* **5-min Double Tops** (already tagged profit 159.37 / 159.79) keep a soft lid on rallies < 161.21 .
* **15-min Triple Top** projects → 154.14 but is still far from activation price; treat as tail-risk only for now .
* **Daily three-black-crows** (historic) keeps medium-term trend neutral-to-down .

Measured-move arithmetic plus ATR(14)=3.1 puts an expanded intraday range at 157.8–161.5, well inside Finnhub’s 157.94–162.20 envelope .

---

**Indicator / divergence snapshot**

* ADX 27–30 across 1-, 5-, 60-min frames ⇒ trending but not extreme .
* Hourly MACD histogram printed three sequential higher lows while price stalled – regular bullish divergence; biasing forecasts slightly upward.
* Volatility regime: 14-ATR / 50-ATR ≈ 0.82 (low-normal) → use 65 % ATR for stop/target calibration.
* Finnhub daily model errs +0.12 \$ on average (30-day MAE = 0.43) – applied as conservative up-side adjustment to close forecast.&#x20;

---

**Trade set-ups**

1. **Dip-buy 159.30 – 159.60**
   *Trigger*: bullish reversal candle on 1-min with RSI < 30 at 159.56 neckline.
   *Stop*: 159.05 (just under DB stop).
   *Target*: 160.48 first, trail to 160.82.
   *Edge*: aligns with incomplete double-bottom and mid-band support.

2. **Breakout continuation > 160.82**
   *Entry*: first 1-min pull-back that holds above 160.82 after a 5-min close > 161.00.
   *Stop*: 160.48.
   *Targets*: 161.20 pivot, then 161.50 (forecast high).
   Monitor tape for large-lot absorption around 161.2 – failure suggests fade.

3. **Fade extremes 161.20–161.50**
   *Condition*: 3-tick rejection on 1-min plus bearish MACD cross.
   *Stop*: 161.70 (just over pattern stop-loss).
   *Targets*: trail to 160.82 / 160.50; full exit 160.18.
   *Probability*: \~45 % given opposing daily up-bias, so size ½ normal.

ATR-calibrated risk (0.45 \$ stops) with 1 % account risk → position ≈ 22 % of base lot.

---

**Risk & money-management notes**

* Size down after 12 : 30 ET when liquidity thins.
* Avoid holding shorts overnight – weekly bias remains mildly up (close projection 160.8).
* If price pierces 157.8, abandon longs; re-evaluate toward 156.3 weekly SR.

---

### SECTION 3 — intraday execution plan

* **Pre-Open (< 09 : 30 ET)**
  • Mark 159.56 / 160.18 / 160.82 / 161.21.
  • No position. *AI position: flat*

* **Open Range (09 : 30 – 10 : 00)**
  • Buy dip 159.30-159.60 if momentum turns.
  • ORB long if 5-min candle breaks 160.82 upper-band.
  *AI position: long if either trigger, else flat*

* **Mid-Morning (10 : 00 – 12 : 00)**
  • Scale ½ at 160.48, move stop to breakeven.
  • Add on PB ≥ 160.82 aiming 161.20.
  *AI position: long / flat*

* **Lunch lull (12 : 00 – 14 : 00)**
  • Reduce exposure by 30 %; tighten stop 160.50.
  • Skip fresh entries unless price compresses around 160.18 pivot.
  *AI position: trimmed long / flat*

* **Afternoon push (14 : 00 – 15 : 30)**
  • If 161.20 prints, initiate tactical short (½ risk) on reversal signal.
  • Manage residual longs: exit at 161.20 or trail 20-EMA(5-min).
  *AI position: hedge-short or flat*

* **Power Hour / Close (15 : 30 – 16 : 00)**
  • Cover shorts by 15 : 55; flat into closing auction regardless of P\&L.
  • End-of-day check: settle > 161.2 flips weekly bias bullish; < 158.8 warns 154-handle test.
  *AI position: flat*
