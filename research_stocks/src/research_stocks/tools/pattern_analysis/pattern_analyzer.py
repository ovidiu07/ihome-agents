# pattern_analyzer.py
# -----------------
# Core pattern analysis functions

import pandas as pd
from typing import List, Dict, Any

from .candlestick_patterns import detect_candlestick_patterns
from .chart_patterns import (
    detect_pivots,
    detect_head_shoulders_pivot,
    detect_double_tops_bottoms_pivot,
    detect_triangle_pivot
)
from .utils import get_pattern_reliability


def calculate_pattern_score(pattern: dict, df: pd.DataFrame, volume_col: str = None) -> float:
    """
    Calculate a score for a pattern based on various factors.
    
    Args:
        pattern: Pattern dictionary
        df: DataFrame with OHLC data
        volume_col: Name of volume column if available
        
    Returns:
        Score value
    """
    # Base score from pattern reliability
    base_score = get_pattern_reliability(pattern['pattern'])
    
    # Pattern height factor (taller patterns are more significant)
    height_factor = pattern.get('height', 0) / df['Close'].mean() * 10
    
    # Duration factor (longer patterns are more significant)
    start_date = pd.to_datetime(pattern['start_date'])
    end_date = pd.to_datetime(pattern['end_date'])
    duration = (end_date - start_date).days + 1
    duration_factor = min(duration / 10, 1.5)  # Cap at 1.5
    
    # Volume factor if volume data is available
    volume_factor = 1.0
    if volume_col and volume_col in df.columns:
        # Calculate average volume during pattern vs. before pattern
        pattern_idx = df[(df['Date'] >= pattern['start_date']) & 
                         (df['Date'] <= pattern['end_date'])].index
        
        if len(pattern_idx) > 0:
            before_idx = df[df.index < pattern_idx[0]].index[-min(20, len(df[df.index < pattern_idx[0]])):]
            
            if len(before_idx) > 0:
                avg_vol_pattern = df.loc[pattern_idx, volume_col].mean()
                avg_vol_before = df.loc[before_idx, volume_col].mean()
                
                if avg_vol_before > 0:
                    volume_factor = min(avg_vol_pattern / avg_vol_before, 2.0)  # Cap at 2.0
    
    # Calculate final score
    score = base_score * (1 + height_factor) * duration_factor * volume_factor
    
    # Normalize to a reasonable range (0-5)
    score = min(max(score, 0), 5)
    
    return score


def resolve_conflicts(patterns: List[Dict]) -> List[Dict]:
    """
    Resolve conflicts between overlapping patterns by keeping the highest-scoring one.
    
    Args:
        patterns: List of pattern dictionaries
        
    Returns:
        List of patterns with conflicts resolved
    """
    if not patterns:
        return patterns
    
    # Sort patterns by score (highest first)
    sorted_patterns = sorted(patterns, key=lambda p: p.get('value', 0), reverse=True)
    
    # Keep track of which patterns to keep
    keep = [True] * len(sorted_patterns)
    
    # Check each pattern against higher-scoring patterns
    for i in range(1, len(sorted_patterns)):
        if not keep[i]:
            continue
            
        p1 = sorted_patterns[i]
        p1_start = pd.to_datetime(p1['start_date'])
        p1_end = pd.to_datetime(p1['end_date'])
        
        for j in range(i):
            if not keep[j]:
                continue
                
            p2 = sorted_patterns[j]
            p2_start = pd.to_datetime(p2['start_date'])
            p2_end = pd.to_datetime(p2['end_date'])
            
            # Check for significant overlap
            overlap_start = max(p1_start, p2_start)
            overlap_end = min(p1_end, p2_end)
            
            if overlap_start <= overlap_end:
                # Calculate overlap percentage
                p1_days = (p1_end - p1_start).days + 1
                overlap_days = (overlap_end - overlap_start).days + 1
                
                if overlap_days / p1_days > 0.5:  # More than 50% overlap
                    keep[i] = False
                    break
    
    # Return only the patterns to keep
    return [p for i, p in enumerate(sorted_patterns) if keep[i]]


def analyze_patterns(df: pd.DataFrame, window: int = 5, volume_col: str = None) -> Dict[str, Any]:
    """
    Analyze price data for various patterns.
    
    Args:
        df: DataFrame with OHLC data
        window: Window size for pattern detection
        volume_col: Name of volume column if available
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        'patterns': [],
        'next_prediction': None
    }
    
    # Ensure we have enough data
    if len(df) < window + 5:
        return results
    
    # Detect pivot points
    pivots = detect_pivots(df)
    
    # Detect candlestick patterns
    candlestick_patterns = detect_candlestick_patterns(df)
    
    # Convert candlestick patterns to our standard format
    for pattern_name in candlestick_patterns.columns:
        for i in range(len(candlestick_patterns)):
            if candlestick_patterns.iloc[i][pattern_name]:
                # Determine pattern direction
                direction = 'bullish'
                if pattern_name in ['Shooting Star', 'Hanging Man', 'Three Black Crows']:
                    direction = 'bearish'
                
                pattern = {
                    'pattern': pattern_name,
                    'start_date': df.iloc[max(0, i-2)]['Date'],  # Start a few bars before
                    'end_date': df.iloc[i]['Date'],
                    'direction': direction,
                    'value': 1.0,  # Default value, will be refined
                    'status': 'Confirmed'
                }
                
                # Calculate pattern height
                if i > 0:
                    height = abs(df.iloc[i]['Close'] - df.iloc[i-1]['Close'])
                    pattern['height'] = height
                
                # Calculate pattern score
                pattern['value'] = calculate_pattern_score(pattern, df, volume_col)
                
                results['patterns'].append(pattern)
    
    # Detect chart patterns
    chart_patterns = []
    chart_patterns.extend(detect_head_shoulders_pivot(df, pivots))
    chart_patterns.extend(detect_double_tops_bottoms_pivot(df, pivots))
    chart_patterns.extend(detect_triangle_pivot(df, pivots, window))
    
    # Calculate scores for chart patterns
    for pattern in chart_patterns:
        pattern['value'] = calculate_pattern_score(pattern, df, volume_col)
    
    # Add chart patterns to results
    results['patterns'].extend(chart_patterns)
    
    # Resolve conflicts between patterns
    results['patterns'] = resolve_conflicts(results['patterns'])
    
    # Sort patterns by date
    results['patterns'] = sorted(results['patterns'], key=lambda p: pd.to_datetime(p['start_date']))
    
    return results