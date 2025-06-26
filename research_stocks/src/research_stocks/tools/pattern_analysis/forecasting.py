# forecasting.py
# -------------
# Functions for forecasting based on pattern analysis

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def _breakout_confirmed(df: pd.DataFrame, pattern: Dict) -> bool:
    """
    Check if a pattern breakout is confirmed by subsequent price action.
    
    Args:
        df: DataFrame with OHLC data
        pattern: Pattern dictionary
        
    Returns:
        Boolean indicating if breakout is confirmed
    """
    # Find the end date of the pattern in the dataframe
    try:
        end_idx = df[df['Date'] == pattern['end_date']].index[0]
    except (IndexError, KeyError):
        return False
    
    # Skip if we're at the end of the dataframe
    if end_idx >= len(df) - 1:
        return False
    
    # Get the next few bars after pattern completion
    next_bars = df.iloc[end_idx+1:min(end_idx+6, len(df))]
    
    if next_bars.empty:
        return False
    
    # Check for breakout confirmation based on pattern direction
    if pattern['direction'] == 'bullish':
        # For bullish patterns, check if price moves above the pattern high
        pattern_high = df.loc[df['Date'] == pattern['end_date'], 'High'].iloc[0]
        return any(next_bars['High'] > pattern_high * 1.01)  # 1% above pattern high
    
    elif pattern['direction'] == 'bearish':
        # For bearish patterns, check if price moves below the pattern low
        pattern_low = df.loc[df['Date'] == pattern['end_date'], 'Low'].iloc[0]
        return any(next_bars['Low'] < pattern_low * 0.99)  # 1% below pattern low
    
    return False


def _label_patterns(df: pd.DataFrame, patterns: List[Dict]) -> pd.DataFrame:
    """
    Add pattern labels to the dataframe.
    
    Args:
        df: DataFrame with OHLC data
        patterns: List of pattern dictionaries
        
    Returns:
        DataFrame with pattern labels
    """
    # Create a copy of the dataframe
    labeled_df = df.copy()
    
    # Add pattern columns
    labeled_df['Pattern'] = ''
    labeled_df['Pattern_Direction'] = ''
    labeled_df['Pattern_Score'] = 0.0
    
    # Label each pattern
    for pattern in patterns:
        try:
            start_idx = labeled_df[labeled_df['Date'] == pattern['start_date']].index[0]
            end_idx = labeled_df[labeled_df['Date'] == pattern['end_date']].index[0]
            
            # Label all rows in the pattern range
            for idx in range(start_idx, end_idx + 1):
                if labeled_df.loc[idx, 'Pattern'] == '':
                    labeled_df.loc[idx, 'Pattern'] = pattern['pattern']
                    labeled_df.loc[idx, 'Pattern_Direction'] = pattern['direction']
                    labeled_df.loc[idx, 'Pattern_Score'] = pattern.get('value', 0)
                else:
                    # If there's already a pattern, append the new one
                    labeled_df.loc[idx, 'Pattern'] += f", {pattern['pattern']}"
                    labeled_df.loc[idx, 'Pattern_Direction'] += f", {pattern['direction']}"
                    labeled_df.loc[idx, 'Pattern_Score'] = max(
                        labeled_df.loc[idx, 'Pattern_Score'], 
                        pattern.get('value', 0)
                    )
        except (IndexError, KeyError):
            continue
    
    return labeled_df


def refine_next_predictions(results: Dict[str, Any], df: pd.DataFrame, days: int = 1, 
                           weight_pattern: float = 0.3, weight_volatility: float = 0.7) -> Dict[str, Any]:
    """
    Refine predictions for the next period based on pattern analysis.
    
    Args:
        results: Dictionary with analysis results
        df: DataFrame with OHLC data
        days: Number of days to forecast
        weight_pattern: Weight for pattern-based prediction
        weight_volatility: Weight for volatility-based prediction
        
    Returns:
        Updated results dictionary with refined predictions
    """
    # Create a copy of results to avoid modifying the original
    refined_results = results.copy()
    
    # If no patterns or not enough data, return original results
    if not results.get('patterns') or len(df) < 30:
        return refined_results
    
    # Get the latest data point
    latest = df.iloc[-1]
    
    # Calculate recent volatility (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # Initialize prediction
    prediction = {
        'direction': 'neutral',
        'confidence': 0.5,
        'O': latest['Open'],
        'H': latest['High'],
        'L': latest['Low'],
        'C': latest['Close']
    }
    
    # Count bullish and bearish patterns
    patterns = results.get('patterns', [])
    recent_patterns = [p for p in patterns if pd.to_datetime(p['end_date']) >= 
                      pd.to_datetime(df.iloc[-10]['Date'])]
    
    bullish_count = sum(1 for p in recent_patterns if p['direction'] == 'bullish')
    bearish_count = sum(1 for p in recent_patterns if p['direction'] == 'bearish')
    
    # Determine direction based on pattern counts
    if bullish_count > bearish_count:
        pattern_direction = 'bullish'
        pattern_confidence = min(0.5 + (bullish_count - bearish_count) * 0.1, 0.9)
    elif bearish_count > bullish_count:
        pattern_direction = 'bearish'
        pattern_confidence = min(0.5 + (bearish_count - bullish_count) * 0.1, 0.9)
    else:
        pattern_direction = 'neutral'
        pattern_confidence = 0.5
    
    # Calculate volatility-based price ranges
    volatility_high = latest['Close'] + atr
    volatility_low = latest['Close'] - atr
    
    # Blend pattern and volatility predictions
    if pattern_direction == 'bullish':
        prediction['direction'] = 'bullish'
        prediction['confidence'] = pattern_confidence
        prediction['H'] = latest['Close'] + atr * (1 + pattern_confidence * 0.5)
        prediction['L'] = max(latest['Close'] - atr * 0.5, volatility_low)
        prediction['C'] = latest['Close'] + (prediction['H'] - latest['Close']) * pattern_confidence
    elif pattern_direction == 'bearish':
        prediction['direction'] = 'bearish'
        prediction['confidence'] = pattern_confidence
        prediction['H'] = min(latest['Close'] + atr * 0.5, volatility_high)
        prediction['L'] = latest['Close'] - atr * (1 + pattern_confidence * 0.5)
        prediction['C'] = latest['Close'] - (latest['Close'] - prediction['L']) * pattern_confidence
    else:
        # Neutral case - use volatility-based range
        prediction['H'] = volatility_high
        prediction['L'] = volatility_low
        prediction['C'] = latest['Close']
    
    # Set the next day's open near the previous close
    prediction['O'] = latest['Close'] * (1 + np.random.normal(0, 0.005))
    
    # Update results with prediction
    refined_results['next_prediction'] = prediction
    
    return refined_results


class _OHLCForecaster:
    """
    Internal class for OHLC forecasting using statistical methods.
    """
    def __init__(self, opens, highs, lows, closes):
        """
        Initialize the forecaster with historical OHLC data.
        
        Args:
            opens: Series of open prices
            highs: Series of high prices
            lows: Series of low prices
            closes: Series of close prices
        """
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        
        # Calculate returns and ranges
        self.close_returns = closes.pct_change().dropna()
        self.high_low_ranges = (highs - lows) / closes
        self.open_close_ranges = abs(opens - closes) / closes
        
        # Calculate statistics
        self.mean_return = self.close_returns.mean()
        self.std_return = self.close_returns.std()
        self.mean_hl_range = self.high_low_ranges.mean()
        self.mean_oc_range = self.open_close_ranges.mean()
        
        # Calculate correlations
        self.corr_matrix = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes
        }).corr()
    
    def predict(self, days=1):
        """
        Generate OHLC predictions for the specified number of days.
        
        Args:
            days: Number of days to forecast
            
        Returns:
            DataFrame with predicted OHLC values
        """
        predictions = []
        last_close = self.closes.iloc[-1]
        
        for _ in range(days):
            # Predict close using random return from normal distribution
            close_return = np.random.normal(self.mean_return, self.std_return)
            pred_close = last_close * (1 + close_return)
            
            # Predict high-low range
            hl_range = np.random.normal(self.mean_hl_range, self.high_low_ranges.std())
            hl_range = max(0.005, hl_range)  # Ensure positive range
            
            # Predict open-close range
            oc_range = np.random.normal(self.mean_oc_range, self.open_close_ranges.std())
            oc_range = max(0.001, oc_range)  # Ensure positive range
            
            # Determine if open is above or below close
            if np.random.random() > 0.5:
                # Bullish day (close > open)
                pred_open = pred_close / (1 + oc_range)
                
                # High is above both open and close
                pred_high = pred_close * (1 + hl_range / 2)
                
                # Low is below open
                pred_low = pred_open * (1 - hl_range / 2)
            else:
                # Bearish day (open > close)
                pred_open = pred_close * (1 + oc_range)
                
                # High is above open
                pred_high = pred_open * (1 + hl_range / 2)
                
                # Low is below close
                pred_low = pred_close * (1 - hl_range / 2)
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            pred_high = max(pred_high, pred_open, pred_close)
            pred_low = min(pred_low, pred_open, pred_close)
            
            # Add prediction
            predictions.append({
                'O': pred_open,
                'H': pred_high,
                'L': pred_low,
                'C': pred_close
            })
            
            # Update last close for next iteration
            last_close = pred_close
        
        return predictions[0] if days == 1 else predictions


def build_feature_stack(df_hist: pd.DataFrame, df_today_min: Optional[pd.DataFrame], 
                       daily_patterns: List[Dict], intraday_patterns: List[Dict], 
                       vwap_trend: str, morning_cutoff: str = "10:30") -> Dict[str, Any]:
    """
    Build a feature stack for forecasting.
    
    Args:
        df_hist: DataFrame with historical daily OHLC data
        df_today_min: DataFrame with today's intraday OHLC data (can be None)
        daily_patterns: List of daily pattern dictionaries
        intraday_patterns: List of intraday pattern dictionaries
        vwap_trend: VWAP trend direction ("UP" or "DOWN")
        morning_cutoff: Time cutoff for morning session
        
    Returns:
        Dictionary with features for forecasting
    """
    features = {}
    
    # Helper function to calculate bias score from patterns
    def _bias_score(patts: List[Dict]) -> float:
        if not patts:
            return 0.0
        
        bullish = sum(1 for p in patts if p['direction'] == 'bullish')
        bearish = sum(1 for p in patts if p['direction'] == 'bearish')
        
        if bullish == bearish:
            return 0.0
        
        # Calculate normalized score between -1 and 1
        total = bullish + bearish
        return (bullish - bearish) / total
    
    # Historical features
    if not df_hist.empty:
        # Price momentum
        features['price_momentum'] = df_hist['Close'].pct_change(5).iloc[-1]
        
        # Volatility
        features['volatility'] = df_hist['High'].sub(df_hist['Low']).div(df_hist['Close']).rolling(10).mean().iloc[-1]
        
        # Daily pattern bias
        features['daily_pattern_bias'] = _bias_score(daily_patterns)
        
        # Recent performance
        features['week_return'] = df_hist['Close'].pct_change(5).iloc[-1]
        features['month_return'] = df_hist['Close'].pct_change(20).iloc[-1]
    
    # Intraday features
    if df_today_min is not None and not df_today_min.empty:
        # Morning vs. full day performance
        morning_data = df_today_min[df_today_min['Date'].str.contains(morning_cutoff, regex=False)]
        
        if not morning_data.empty:
            morning_open = morning_data.iloc[0]['Open']
            morning_close = morning_data.iloc[-1]['Close']
            features['morning_return'] = (morning_close / morning_open) - 1
        
        # Intraday pattern bias
        features['intraday_pattern_bias'] = _bias_score(intraday_patterns)
        
        # Intraday volatility
        features['intraday_volatility'] = df_today_min['High'].max() / df_today_min['Low'].min() - 1
        
        # VWAP trend
        features['vwap_trend'] = 1 if vwap_trend == "UP" else -1
    
    # Combined features
    features['combined_bias'] = (
        features.get('daily_pattern_bias', 0) * 0.6 + 
        features.get('intraday_pattern_bias', 0) * 0.4
    )
    
    return features


def probabilistic_day_forecast(features: Dict[str, Any], open_price: float, 
                              base_prob: float = 0.50, alpha: float = 4.0) -> Dict[str, Any]:
    """
    Generate a probabilistic forecast for the day based on features.
    
    Args:
        features: Dictionary with features for forecasting
        open_price: Opening price for the day
        base_prob: Base probability for direction
        alpha: Scaling factor for confidence
        
    Returns:
        Dictionary with forecast information
    """
    # Calculate direction probability
    combined_bias = features.get('combined_bias', 0)
    vwap_trend = features.get('vwap_trend', 0)
    
    # Blend biases with weights
    weighted_bias = combined_bias * 0.7 + vwap_trend * 0.3
    
    # Convert to probability (sigmoid-like transformation)
    direction_prob = base_prob + (1 - base_prob) * np.tanh(alpha * weighted_bias)
    
    # Determine direction
    if direction_prob > 0.55:
        direction = "bullish"
    elif direction_prob < 0.45:
        direction = "bearish"
    else:
        direction = "neutral"
    
    # Calculate confidence
    confidence = abs(direction_prob - 0.5) * 2  # Scale to 0-1
    
    # Estimate volatility
    volatility = features.get('volatility', 0.015)  # Default to 1.5% if not available
    intraday_vol = features.get('intraday_volatility', volatility)
    blended_vol = (volatility + intraday_vol) / 2
    
    # Calculate price targets
    if direction == "bullish":
        high = open_price * (1 + blended_vol * (1 + confidence))
        low = open_price * (1 - blended_vol * (1 - confidence))
        close = open_price * (1 + blended_vol * confidence)
    elif direction == "bearish":
        high = open_price * (1 + blended_vol * (1 - confidence))
        low = open_price * (1 - blended_vol * (1 + confidence))
        close = open_price * (1 - blended_vol * confidence)
    else:
        high = open_price * (1 + blended_vol)
        low = open_price * (1 - blended_vol)
        close = open_price
    
    # Return forecast
    return {
        'direction': direction,
        'confidence': confidence,
        'O': open_price,
        'H': high,
        'L': low,
        'C': close
    }