import json
from datetime import datetime

import pandas as pd

from research_stocks.tools.pattern_analysis.reporting import export_enhanced_results


def test_export_enhanced_results_datetime(tmp_path):
    df = pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=2, freq="T"),
        "Open": [1.0, 2.0],
        "Close": [1.5, 2.5],
    })
    results = {"symbol": "TST", "intraday": df}
    export_enhanced_results(results, output_dir=str(tmp_path))

    today = datetime.now().strftime("%d-%m-%Y")
    file_path = tmp_path / today / f"TST_Json_{today.split('-')[0]}{today.split('-')[1]}"
    with open(file_path) as f:
        data = json.load(f)
    assert isinstance(data["intraday"][0]["Datetime"], str)

