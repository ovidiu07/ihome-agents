from crewai.tools.base_tool import BaseTool

class OHLCFormatterTool(BaseTool):
  name: str = "OHLCFormatterTool"
  description: str = "Format OHLC data into a short readable summary for LLMs."

  def _run(self, ohlc_data: list[dict], symbol: str = "UNKNOWN", max_rows: int = 5) -> str:
    if not ohlc_data or not isinstance(ohlc_data, list):
      return f"No OHLC data found for {symbol}."

    recent = ohlc_data[-max_rows:]
    lines = [f"**{symbol.upper()}** last {len(recent)} candles:"]
    for entry in reversed(recent):
      date = entry.get("date", "???")
      open_ = entry.get("open", "N/A")
      high = entry.get("high", "N/A")
      low = entry.get("low", "N/A")
      close = entry.get("close", "N/A")
      volume = entry.get("volume", "N/A")
      lines.append(f"- {date}: O={open_} H={high} L={low} C={close} Vol={volume}")

    return "\n".join(lines)