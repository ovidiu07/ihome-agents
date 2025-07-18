**Section 1**

```json
{
  "symbol": "TSLA",
  "forecast_generated_utc": "2025-07-16T13:02:12Z",
  "overall_bias": "Neutral-to-slightly-bearish intraday; broadly neutral on the daily/weekly charts",
  "timeframes": {
    "1min": {
      "bias": "UP (micro-bounce)",
      "confidence": 1.0,
      "expected_move": {
        "open": 309.761,
        "high": 309.861,
        "low": 309.741,
        "close": 309.811
      },
      "key_supports": [308.989],
      "key_resistances": [309.871],
      "notable_patterns": ["Triple Top (bearish, complete)"],
      "risk_note": "Long scalp only while > 309.74; pattern failure above 309.87 invalidates bearish setup."
    },

    "15min": {
      "bias": "UP → range-cap",
      "confidence": 1.0,
      "expected_move": {
        "open": 309.641,
        "high": 310.153,
        "low": 309.539,
        "close": 309.897
      },
      "key_supports": [307.989, 310.489],
      "key_resistances": [312.391, 315.501],
      "notable_patterns": ["Wedge (bearish, complete)", "Three Black Crows (bearish, complete)"],
      "risk_note": "Upside target sits well below first resistance (310.49) — watch for fade if buyers stall there."
    },

    "1h": {
      "bias": "DOWN",
      "confidence": 1.0,
      "expected_move": {
        "open": 311.471,
        "high": 311.934,
        "low": 309.155,
        "close": 310.313
      },
      "key_supports": [305.650, 297.820],
      "key_resistances": [314.090, 322.590],
      "notable_patterns": ["Double Top (bearish, complete)"],
      "risk_note": "Hourly bias conflicts with shorter-term bounce; probability of lower highs under 314.09."
    },

    "1d": {
      "bias": "SIDEWAYS / slightly bearish",
      "confidence": 0.079,
      "expected_move": {
        "open": 310.78,
        "high": 314.254,
        "low": 307.306,
        "close": 310.78
      },
      "key_supports": [300.000, 243.360],
      "key_resistances": [318.450, 377.290],
      "notable_patterns": ["Triangle (bearish, complete & incomplete variants)"],
      "risk_note": "Lack of trend (ADX ≈ 18) — expect chop until a daily close outside 307 – 315."
    },

    "1w": {
      "bias": "SIDEWAYS",
      "confidence": 0.022,
      "expected_move": {
        "open": 316.90,
        "high": 321.120,
        "low": 312.680,
        "close": 316.90
      },
      "key_supports": [312.680, 240.700],
      "key_resistances": [326.200, 488.540],
      "notable_patterns": ["Double Top (bearish, incomplete)"],
      "risk_note": "Weekly failure to clear 326 keeps longer-term sellers in control; break below 312.68 opens 300/288."
    }
  }
}
```

---

**Section 2**
*Key Take-aways & How the Forecast Was Enhanced*

1. **Pattern Confluence.**

    * Short-timeframes (1 min & 15 min) show *completed* bearish formations (Triple Top, Wedge, Three-Black-Crows) yet the algorithmic component-forecasts point marginally higher. I kept the *UP* bias but tempered it with tight resistance levels and explicit “fade/invalid” notes to reflect that contradiction.

2. **Support/Resistance Integration.**

    * Every timeframe pulls the closest SR pairs from its own `support_resistance.levels` array and checks them against the algorithm’s projected high/low range. Where projection < nearest resistance, I flag “range-cap”; where projection hovers just above support, I add “long scalp only while > support.”

3. **Trend Strength via ADX.**

    * Only the 5 min/15 min sets have ADX > 25 (trending); hourly and higher remain non-trending. Section 1 bias labels reflect this (e.g., “range-cap”, “SIDEWAYS”).

4. **Risk Notes.**

    * For each timeframe I append a concise risk clause: what invalidates the call, or what would confirm it. This is the “enhancement” beyond the raw point forecast.

5. **Holistic Bias.**

    * The “overall\_bias” line synthesises the stack: micro-bounce possible, but with hourly pressure and unresolved daily triangle, traders should treat any intraday strength as tactical until > 314–315 or < 307 breaks.

> **Disclaimers:**
> • Forecasts are probabilistic, not guarantees.
> • Use proper risk management; past patterns do not assure future results.
> • This is informational only, not investment advice.
