# main.py
# -------
# Main execution logic for pattern analysis

import os
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to allow importing from sibling modules
sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv

# Import from our new modules
from .data_fetchers import fetch_intraday_bars, fetch_daily_history
from .pattern_filters import (
    drop_duplicates, 
    suppress_nearby_hits, 
    cluster_and_keep_best, 
    filter_patterns_by_criteria,
    remove_duplicates_by_status
)
from .forecast_utils import (
    get_intraday_bias, 
    get_daily_bias, 
    blended_forecast, 
    calculate_vwap_obv_trend,
    calculate_atr
)

# Import from advanced_pattern_helpers (these will be refactored later)
from advanced_pattern_helpers import (
    analyze_patterns, 
    export_analysis_results,
    print_summary_report,
    refine_next_predictions,
    generate_evolving_daily_ohlc,
    build_feature_stack,
    probabilistic_day_forecast
)


def main() -> None:
    """Main function to run pattern analysis."""
    global df_today
    load_dotenv()
    poly_key = os.getenv("POLYGON_KEY")
    if not poly_key:
        print("❌  POLYGON_KEY not set in environment variables.")
        return

    # ── Daily history (12 months minus current session) ──
    df_hist = fetch_daily_history("TSLA", period="12mo")

    # ── Intraday bars (today) ──
    df_today_min = fetch_intraday_bars("TSLA", poly_key, limit=150)

    if df_today_min is None or df_today_min.empty:
        print("⚠️  Skipping intraday pattern scan — no data.")
        df_combined = df_hist.tail(180)
        intraday_filtered = []
    else:
        print("\n🔍 Running pattern analysis on earliest data for today…")

        # ── 1‑minute pattern scan (20‑bar window, strict filters) ──
        raw_intraday = analyze_patterns(df_today_min, window=20)["patterns"]

        # Filter patterns based on criteria
        intraday_filtered = filter_patterns_by_criteria(
            raw_intraday, 
            min_value=1.2, 
            status="Confirmed", 
            min_duration_minutes=20
        )
        intraday_filtered = suppress_nearby_hits(intraday_filtered, gap=10)

        if intraday_filtered:
            print("\n🧠 Premarket / morning pattern summary:")
            print_summary_report({"patterns": intraday_filtered}, show_forecast=False)
        else:
            print("ℹ️  No qualifying intraday patterns.")

        # Aggregate today's intraday to an evolving daily OHLC row
        df_today = generate_evolving_daily_ohlc(df_today_min)
        df_today = pd.DataFrame([df_today])
        df_combined = pd.concat([df_hist.tail(180), df_today], ignore_index=True)

    # ── Full pattern analysis on daily candles ──
    results = analyze_patterns(df_combined, window=5)

    # Process daily patterns
    daily_patterns = drop_duplicates(results["patterns"])
    daily_patterns = remove_duplicates_by_status(daily_patterns, "Duplicate")
    daily_patterns = cluster_and_keep_best(daily_patterns, overlap=0.7)

    # Update results with processed patterns
    results["patterns"] = daily_patterns
    results = refine_next_predictions(results, df_combined)
    export_analysis_results(results)

    # Print daily pattern summary
    if results["patterns"]:
        print("\n📊 Daily pattern summary (look-back 12 mo):")
        for p in results["patterns"]:
            print(f"- {p['start_date']} to {p['end_date']}: "
                  f"{p['pattern']} ({p['direction']}, "
                  f"score={float(p['value']):.2f}, status={p['status']})")
    else:
        print("ℹ️  No patterns found in daily data.")

    # Calculate VWAP and OBV trend
    df_vol = df_today_min if (df_today_min is not None and not df_today_min.empty) else df_hist
    trend = calculate_vwap_obv_trend(df_vol)

    # Build feature stack for forecasting
    if df_today_min is not None and not df_today_min.empty:
        features = build_feature_stack(df_hist, df_today_min, daily_patterns, intraday_filtered, trend)
        today_open = df_today_min.iloc[0]["Open"]
    else:
        features = build_feature_stack(df_hist, None, daily_patterns, intraday_filtered, trend)
        today_open = df_hist.iloc[-1]["Open"]

    # ── Ensemble forecast ──
    atr14 = calculate_atr(df_hist, period=14)
    ensemble = blended_forecast(
        intraday_direction=get_intraday_bias(intraday_filtered),
        daily_direction=get_daily_bias(daily_patterns), 
        vwap_trend=trend,
        atr=atr14
    )

    # Print forecasts and summaries
    print(f"\n🔮 Ensemble forecast: {ensemble}")
    print_summary_report(results, show_forecast=True)
    print("\n🔮 OHLC for today's earliest intraday bars:")
    print(df_today.head(1).to_string(index=False))
    print(f"\n📈 VWAP/OBV trend hint: {trend}")

    day_fcast = probabilistic_day_forecast(features, today_open)
    print(f"\n🔮 Prob-weighted forecast ({day_fcast['direction']}, "
          f"conf {day_fcast['confidence']:.0%}) "
          f"O={day_fcast['O']} H={day_fcast['H']} L={day_fcast['L']} C={day_fcast['C']}")


if __name__ == "__main__":
    main()
