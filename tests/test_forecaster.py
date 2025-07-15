import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research_stocks', 'src'))
from research_stocks.tools.pattern_analysis.forecasting import (
    next_prediction_from_finnhub,
)


SAMPLE_PAYLOAD = {
    "fintech_daily": {
        "resolution": "D",
        "candles": {
            "o": [100.0 + i for i in range(40)],
            "h": [102.0 + i for i in range(40)],
            "l": [99.0 + i for i in range(40)],
            "c": [101.0 + i for i in range(40)],
            "v": [1000 + i for i in range(40)],
            "t": list(range(40)),
        },
    },
    "next_prediction_from_finnhub": {
        "component_forecasts": {
            "1d": {
                "open": 102.0,
                "high": 103.5,
                "low": 100.5,
                "close": 102.5,
                "confidence": 0.8,
                "trend": "UP",
            }
        }
    },
}


def test_forecaster():
    out = next_prediction_from_finnhub(SAMPLE_PAYLOAD)
    assert {
        "direction",
        "prob_up",
        "confidence",
        "open",
        "high",
        "low",
        "close",
        "feature_stack",
    } <= out.keys()
