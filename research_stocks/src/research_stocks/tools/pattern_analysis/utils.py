# utils.py
# --------
# Utility functions for pattern analysis
from __future__ import annotations

import pandas as pd
from typing import List, Dict


def get_pattern_reliability(name: str | None = None):
    """
    Return a reliability score (0-1) for a given pattern *or* the full mapping
    when no name is provided.

    A higher value means the pattern is considered more trustworthy.
    """
    # Subjective values based on common technical-analysis literature
    reliability = {
        "Head and Shoulders": 0.75,
        "Inverse Head and Shoulders": 0.75,
        "Double Top": 0.70,
        "Double Bottom": 0.70,
        "Triple Top": 0.80,
        "Triple Bottom": 0.80,
        "Ascending Triangle": 0.65,
        "Descending Triangle": 0.65,
        "Symmetrical Triangle": 0.60,
        "Rising Wedge": 0.60,
        "Falling Wedge": 0.60,
        "Channel Up": 0.55,
        "Channel Down": 0.55,
    }

    # Legacy behaviour: return the whole dict if no specific pattern requested
    if name is None:
        return reliability

    # Normal behaviour: single pattern lookup
    return reliability.get(name, 0.5)  # Default to 0.5 for unknown patterns


def ensure_pattern_dates_are_datetime(patterns: List[Dict]) -> List[Dict]:
    """
    Ensure that start_date and end_date in patterns are datetime objects.
    This is useful for consistent date handling across the codebase.
    
    Args:
        patterns: List of pattern dictionaries
        
    Returns:
        List of patterns with datetime objects for dates
    """
    for p in patterns:
        if "start_date" in p and not isinstance(p["start_date"], pd.Timestamp):
            p["start_date"] = pd.to_datetime(p["start_date"])
        if "end_date" in p and not isinstance(p["end_date"], pd.Timestamp):
            p["end_date"] = pd.to_datetime(p["end_date"])
    return patterns


def slope(series: pd.Series) -> float:
    """
    Calculate the slope of a series.
    
    Args:
        series: Pandas Series of values
        
    Returns:
        Slope value
    """
    # Create a series of x values (0, 1, 2, ...)
    x = pd.Series(range(len(series)))
    
    # Calculate the slope using the formula:
    # slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    n = len(series)
    if n < 2:
        return 0
    
    xy_sum = (x * series).sum()
    x_sum = x.sum()
    y_sum = series.sum()
    x_squared_sum = (x ** 2).sum()
    
    denominator = n * x_squared_sum - x_sum ** 2
    if denominator == 0:
        return 0
    
    return (n * xy_sum - x_sum * y_sum) / denominator