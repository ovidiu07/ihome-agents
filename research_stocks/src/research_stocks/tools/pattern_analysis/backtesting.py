# backtesting.py
# -------------
# Functions for backtesting pattern strategies

import pandas as pd
from typing import List, Dict


def backtest_pattern_strategy(df: pd.DataFrame, patterns: List[Dict], risk: float = 0.01) -> Dict:
    """
    Backtest a trading strategy based on detected patterns.
    
    Args:
        df: DataFrame with OHLC data
        patterns: List of pattern dictionaries
        risk: Risk per trade as a fraction of capital
        
    Returns:
        Dictionary with backtest results
    """
    # Initialize results
    results = {
        'trades': [],
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'total_return': 0.0,
        'max_drawdown': 0.0
    }
    
    # Initialize account
    capital = 10000.0  # Starting capital
    equity = [capital]
    
    # Track trades
    trades = []
    
    # Process each pattern
    for pattern in patterns:
        # Skip patterns without a clear direction
        if 'direction' not in pattern or pattern['direction'] == 'neutral':
            continue
        
        # Find the end date of the pattern in the dataframe
        try:
            end_idx = df[df['Date'] == pattern['end_date']].index[0]
        except (IndexError, KeyError):
            continue
        
        # Skip if we're at the end of the dataframe
        if end_idx >= len(df) - 1:
            continue
        
        # Entry is the next day after pattern completion
        entry_idx = end_idx + 1
        entry_price = df.iloc[entry_idx]['Open']
        
        # Determine stop loss and take profit levels
        pattern_height = pattern.get('height', 0)
        if pattern_height == 0:
            # If height not provided, use ATR or a percentage of price
            pattern_height = entry_price * 0.02  # 2% of price
        
        if pattern['direction'] == 'bullish':
            stop_loss = entry_price - pattern_height
            take_profit = entry_price + pattern_height * 2  # 2:1 reward-to-risk
        else:  # bearish
            stop_loss = entry_price + pattern_height
            take_profit = entry_price - pattern_height * 2  # 2:1 reward-to-risk
        
        # Calculate position size based on risk
        risk_amount = capital * risk
        position_size = risk_amount / abs(entry_price - stop_loss)
        
        # Simulate the trade
        exit_idx = None
        exit_price = None
        exit_reason = None
        
        for i in range(entry_idx + 1, min(entry_idx + 20, len(df))):  # Look ahead up to 20 bars
            high = df.iloc[i]['High']
            low = df.iloc[i]['Low']
            
            if pattern['direction'] == 'bullish':
                if low <= stop_loss:
                    exit_idx = i
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                    break
                elif high >= take_profit:
                    exit_idx = i
                    exit_price = take_profit
                    exit_reason = 'Take Profit'
                    break
            else:  # bearish
                if high >= stop_loss:
                    exit_idx = i
                    exit_price = stop_loss
                    exit_reason = 'Stop Loss'
                    break
                elif low <= take_profit:
                    exit_idx = i
                    exit_price = take_profit
                    exit_reason = 'Take Profit'
                    break
        
        # If no exit triggered, use the close of the last bar
        if exit_idx is None:
            exit_idx = min(entry_idx + 19, len(df) - 1)
            exit_price = df.iloc[exit_idx]['Close']
            exit_reason = 'Time Exit'
        
        # Calculate trade result
        if pattern['direction'] == 'bullish':
            profit = (exit_price - entry_price) * position_size
        else:  # bearish
            profit = (entry_price - exit_price) * position_size
        
        # Update capital
        capital += profit
        equity.append(capital)
        
        # Record the trade
        trade = {
            'pattern': pattern['pattern'],
            'direction': pattern['direction'],
            'entry_date': df.iloc[entry_idx]['Date'],
            'entry_price': entry_price,
            'exit_date': df.iloc[exit_idx]['Date'],
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'profit': profit,
            'return': profit / (capital - profit)
        }
        trades.append(trade)
    
    # Calculate performance metrics
    if trades:
        wins = sum(1 for t in trades if t['profit'] > 0)
        losses = sum(1 for t in trades if t['profit'] <= 0)
        
        results['trades'] = trades
        results['win_rate'] = wins / len(trades) if len(trades) > 0 else 0
        
        total_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        total_loss = abs(sum(t['profit'] for t in trades if t['profit'] <= 0))
        results['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        results['total_return'] = (capital / 10000.0) - 1.0
        
        # Calculate max drawdown
        peak = 10000.0
        drawdown = 0.0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            drawdown = max(drawdown, dd)
        
        results['max_drawdown'] = drawdown
    
    return results