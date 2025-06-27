from __future__ import annotations

from typing import List, Dict

import pandas as pd

from .utils import ensure_pattern_dates_are_datetime


def backtest_pattern_strategy(df: pd.DataFrame, patterns: List[Dict],
    hold_days: int = 1, slippage_bps: float = 1.0, risk_per_trade: float = 0.01,
    initial_capital: float = 10_000.0, ) -> Dict[str, any]:
  """
  Very small, self-contained back-tester:
  • Goes LONG at next day's open when a bullish pattern is active.
  • Goes SHORT at next day's open when a bearish pattern is active.
  • Closes the position `hold_days` later at the close.
  • Position sizing = `risk_per_trade` * equity    (no compounding leverage).
  • One trade at a time   (overlapping signals are ignored).

  Returns
  -------
  dict  with keys:
      equity_curve : pd.Series
      trades       : list[dict]
      total_return : float   (#%)
  """

  if df.empty:
    return {"equity_curve": pd.Series(dtype=float), "trades": [],
            "total_return": 0.0}

  df = df.copy()
  df["Date"] = pd.to_datetime(df["Date"])
  df.set_index("Date", inplace=True)

  patterns = ensure_pattern_dates_are_datetime(patterns)
  trades: List[Dict] = []
  equity = initial_capital
  equity_curve = []

  current_pos: Dict | None = None  # {'side','entry_price','size','exit_date'}

  for date, row in df.iterrows():
    # ---------------- enter new trade -------------------------------
    if current_pos is None:
      todays_patterns = [p for p in patterns if
        p["start_date"] <= date <= p["end_date"]]
      if todays_patterns:
        # pick the highest-reliability pattern
        pat = max(todays_patterns, key=lambda p: p.get("score", 0.5))
        side = +1 if pat["direction"] == "bullish" else -1
        entry_price = row["Open"] * (1 + side * slippage_bps * 1e-4)
        size = (equity * risk_per_trade) / entry_price
        exit_date = date + pd.Timedelta(days=hold_days)
        current_pos = {"side": side, "entry_price": entry_price, "size": size,
          "exit_date": exit_date, "pattern": pat["name"], "entry_date": date, }

    # ---------------- exit due trades -------------------------------
    if current_pos is not None and date >= current_pos["exit_date"]:
      exit_price = row["Close"] * (
            1 - current_pos["side"] * slippage_bps * 1e-4)
      pnl = (exit_price - current_pos["entry_price"]) * current_pos["size"] * \
            current_pos["side"]
      equity += pnl
      trades.append({"pattern": current_pos["pattern"],
        "entry_date": current_pos["entry_date"], "exit_date": date,
        "side": "long" if current_pos["side"] == 1 else "short",
        "entry": current_pos["entry_price"], "exit": exit_price, "pnl": pnl, })
      current_pos = None

    equity_curve.append(equity)

  total_return = (equity / initial_capital - 1.0) * 100.0
  return {"equity_curve": pd.Series(equity_curve,
                                    index=df.index[: len(equity_curve)]),
    "trades": trades, "total_return": round(total_return, 2), }
