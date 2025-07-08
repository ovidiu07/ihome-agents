import pandas as pd
import requests
from research_stocks.tools.pattern_analysis.data_fetchers import (
    fetch_intraday_bars, fetch_polygon_intraday
)

class MockResp:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self._data

def _mock_get(data):
    def _inner(*args, **kwargs):
        return MockResp(data)
    return _inner

def sample_data():
    return {
        "results": [
            {"t": 1704207600000, "o": 1.0, "h": 1.2, "l": 0.8, "c": 1.1, "v": 100}
        ]
    }

def test_fetch_intraday_bars(monkeypatch):
    monkeypatch.setattr(requests, "get", _mock_get(sample_data()))
    df = fetch_intraday_bars("AAPL", "key")
    assert "Date" in df.columns
    assert "Datetime" not in df.columns
    assert df.shape[0] == 1


def test_fetch_polygon_intraday(monkeypatch):
    monkeypatch.setattr(requests, "get", _mock_get(sample_data()))
    df = fetch_polygon_intraday("AAPL", "key", interval=15, days=1)
    assert "Date" in df.columns
    assert "Datetime" not in df.columns
    assert df.shape[0] == 1
