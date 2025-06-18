from tools.market_data_tools import PoliticalNewsTool


if __name__ == "__main__":
  tool = PoliticalNewsTool()
  result = tool.run(query="inflation OR interest rates", days_back=2)
  print("=== Political News Result ===")
  for item in result[:5]:  # Print just the first 5 headlines
    print(f"- {item.get('title')} ({item.get('source', {}).get('name')})")