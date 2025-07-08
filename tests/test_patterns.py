import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research_stocks', 'src'))
from research_stocks.tools.pattern_analysis import candlestick_patterns as cp
from research_stocks.tools.pattern_analysis import chart_patterns as chp

# Bypass volume confirmation for tests
cp._volume_confirm = lambda df, mult=1.2: pd.Series([True] * len(df))


def df_single(open_, high, low, close):
    return pd.DataFrame({
        'Open': [open_],
        'High': [high],
        'Low': [low],
        'Close': [close],
        'Volume': [100]
    })


def test_hammer_positive():
    df = df_single(10,10.3,9,10.2)
    assert isinstance(cp.cs_hammer(df).iloc[0], (bool, np.bool_))

def test_hammer_negative():
    df = df_single(10,10.8,9.7,10.5)
    assert not cp.cs_hammer(df).iloc[0]

def test_inverted_hammer_positive():
    df = df_single(10.1,11,10.05,10.2)
    assert cp.cs_inverted_hammer(df).iloc[0]

def test_shooting_star_positive():
    df = df_single(10.0,11.0,9.7,9.8)
    assert cp.cs_shooting_star(df).iloc[0]

def test_doji_positive():
    df = df_single(10,10.2,9.8,10.01)
    assert isinstance(cp.cs_doji(df).iloc[0], (bool, np.bool_))


def make_three_white_soldiers():
    data = [
        {'Open':9,'High':9.6,'Low':8.8,'Close':9.5},
        {'Open':9.4,'High':10.1,'Low':9.3,'Close':10.0},
        {'Open':9.9,'High':10.6,'Low':9.8,'Close':10.5},
    ]
    return pd.DataFrame(data)

def test_three_white_soldiers_positive():
    df = make_three_white_soldiers()
    res = cp.cs_three_white_soldiers(df)
    assert res.iloc[-1]

def test_three_white_soldiers_negative_gap():
    df = make_three_white_soldiers()
    df.loc[1,'Open'] = 9.7
    res = cp.cs_three_white_soldiers(df)
    assert not res.iloc[-1]


def make_three_black_crows():
    data = [
        {'Open':10.5,'High':10.6,'Low':10.2,'Close':10.0},
        {'Open':10.0,'High':10.1,'Low':9.8,'Close':9.5},
        {'Open':9.6,'High':9.7,'Low':9.5,'Close':9.4},
    ]
    return pd.DataFrame(data)

def test_three_black_crows_positive():
    df = make_three_black_crows()
    res = cp.cs_three_black_crows(df)
    assert res.iloc[-1]


def test_morning_star_positive():
    data=[
        {'Open':10,'High':10.2,'Low':9,'Close':9},
        {'Open':8.8,'High':9.1,'Low':8.7,'Close':8.9},
        {'Open':9.1,'High':9.8,'Low':9.0,'Close':9.7},
    ]
    df=pd.DataFrame(data)
    assert cp.cs_morning_star(df).iloc[-1]


def test_evening_star_negative():
    data=[
        {'Open':9,'High':9.8,'Low':8.9,'Close':9.7},
        {'Open':9.9,'High':10.3,'Low':9.8,'Close':10.0},
        {'Open':9.85,'High':9.96,'Low':9.3,'Close':9.2},
    ]
    df=pd.DataFrame(data)
    assert cp.cs_evening_star(df).iloc[-1]


def test_bullish_harami_positive():
    data=[{'Open':10,'High':10.5,'Low':9.5,'Close':9.5},
          {'Open':9.6,'High':9.8,'Low':9.4,'Close':9.7}]
    df=pd.DataFrame(data)
    assert cp.cs_bullish_harami(df).iloc[-1]


def test_bearish_harami_positive():
    data=[{'Open':9.5,'High':9.8,'Low':9.2,'Close':9.8},
          {'Open':9.7,'High':9.9,'Low':9.6,'Close':9.6}]
    df=pd.DataFrame(data)
    assert cp.cs_bearish_harami(df).iloc[-1]


def test_double_top_positive():
    dates=pd.date_range('2024-01-01', periods=9)
    df=pd.DataFrame({
        'Date':dates,
        'Open':[9,10.9,9.2,8.5,8.8,10.8,9.2,9.1,9.0],
        'High':[9,11,9.5,8.5,9.2,10.5,11.1,9.3,9.2],
        'Low':[8,9.5,8.9,8.2,8.5,9.0,9.6,8.8,8.6],
        'Close':[8.5,10.8,9,8.3,8.9,10.4,10.9,9.1,8.9],
    })
    piv=chp.detect_pivots(df,left=1,right=1)
    pats=chp.detect_double_tops_bottoms_pivot(df,piv)
    assert any(p['pattern']=='Double Top' for p in pats)


def test_double_top_negative():
    dates=pd.date_range('2024-01-01', periods=7)
    df=pd.DataFrame({
        'Date':dates,
        'Open':[9,12,9.2,8.5,8.8,11,9.2],
        'High':[9,12,9.5,8.5,9.2,10.2,9.3],
        'Low':[8,9.5,8.9,8.2,8.5,9.6,8.8],
        'Close':[8.5,11.8,9,8.3,8.9,10.9,9.1],
    })
    piv=chp.detect_pivots(df,left=1,right=1)
    pats=chp.detect_double_tops_bottoms_pivot(df,piv)
    assert not any(p['pattern']=='Double Top' for p in pats)
