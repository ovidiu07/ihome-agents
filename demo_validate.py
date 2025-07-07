import os
import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'research_stocks', 'src'))
from research_stocks.tools.pattern_analysis import pattern_analyzer

START="2024-10-01"
END="2025-07-02"
SYMBOL="NVDA"

def main():
    data=yf.download(SYMBOL,start=START,end=END,progress=False)
    data.reset_index(inplace=True)
    data.rename(columns=str.capitalize,inplace=True)
    df_summary=data.tail(30)
    res=pattern_analyzer.analyze_patterns(SYMBOL,data,df_summary)
    pats=res['patterns'][-15:]
    for p in pats:
        print(f"{p['start_date']} -> {p['end_date']} | {p['pattern']} | {p['value']:.2f}")

if __name__=='__main__':
    main()
