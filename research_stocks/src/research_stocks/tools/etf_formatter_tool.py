from crewai_tools import BaseTool, tool

@tool
class ETFDataFormatterTool(BaseTool):
  name: str = "ETFDataFormatterTool"
  description: str = "Formats raw ETF data into concise summary with price, volume, and key indicators."

  def _run(self, raw_data: dict) -> str:
    if "error" in raw_data:
      return f"⚠️ Error in ETFDataTool: {raw_data['error']}"

    if not isinstance(raw_data, dict) or not raw_data:
      return "⚠️ No valid ETF data received."

    summary_lines = []
    for symbol, data in raw_data.items():
      try:
        price = data["lastTrade"]["p"]
        volume = data["day"]["v"]
        open_ = data["day"]["o"]
        high = data["day"]["h"]
        low = data["day"]["l"]
        close = data["day"]["c"]

        summary_lines.append(
            f"{symbol}: O={open_} H={high} L={low} C={close} | Last={price} | Vol={volume}"
        )
      except Exception as e:
        summary_lines.append(f"{symbol}: ⚠️ Error parsing data: {e}")

    return "\n".join(summary_lines)