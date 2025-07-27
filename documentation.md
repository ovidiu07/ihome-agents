# Stock Analysis Application Documentation

## Overview

This application provides automated stock analysis using financial data from Finnhub and other sources. It allows users to submit stock symbols through a web interface, which then triggers analysis of those stocks at scheduled times. The analysis involves pattern recognition, technical indicators, and other financial data processing.

## Setup and Installation

### Prerequisites

- Docker and Docker Compose installed on your system
- Finnhub API token (sign up at https://finnhub.io/)
- (Optional) AWS credentials for S3 storage

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
FINNHUB_TOKEN=your_finnhub_api_token
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=stock-forecasts
```

### Running the Application

1. Build and start the Docker container:

```bash
docker-compose up -d
```

2. Access the web interface at http://localhost:8000

3. To stop the application:

```bash
docker-compose down
```

## Using the Application

1. Open the web interface at http://localhost:8000
2. Enter stock symbols in the text area (e.g., "AAPL, MSFT, NVDA")
3. Click "Submit" to add these symbols to today's analysis queue
4. The application will automatically analyze these stocks at scheduled times (15:00, 16:00, 17:00, 18:30, and 19:30 Eastern Time)
5. Analysis results are stored in the `output` directory and optionally uploaded to S3

## Core Components

### fintech.py

The `fintech.py` module is responsible for interacting with financial APIs and processing financial data. Key functionalities include:

1. **API Integration**: Connects to Finnhub API to fetch financial data with proper error handling and retries.

2. **Data Models**: Defines Pydantic models for various financial data types:
   - `QuoteResponse`: Real-time stock quotes
   - `CompanyNewsItem`: News articles related to a company
   - `CandleResponse`: OHLCV (Open, High, Low, Close, Volume) candle data
   - `PatternRecognitionResponse`: Detected chart patterns
   - `SupportResistanceResponse`: Support and resistance levels
   - `AggregateIndicatorResponse`: Aggregated technical indicator scores

3. **Data Fetching Functions**:
   - `get_quote()`: Fetches real-time quotes for a symbol
   - `get_company_news()`: Retrieves news articles for a company
   - `get_candles()`: Gets OHLCV candle data with fallback to yfinance
   - `get_pattern_recognition()`: Identifies chart patterns
   - `get_support_resistance()`: Determines support and resistance levels
   - `get_aggregate_indicator()`: Gets aggregated technical indicator scores
   - `get_technical_indicator()`: Retrieves specific technical indicators with optimized parameters

4. **High-Level Functions**:
   - `fetch_all()`: Comprehensive function that fetches and saves all financial data for a symbol

### crew.py

The `crew.py` module orchestrates the stock analysis process using CrewAI, which implements an agent-based workflow. Key functionalities include:

1. **Agent Definitions**:
   - `data_harvester_agent()`: Collects and processes financial data
   - `report_composer_agent()`: Creates analysis reports
   - `forecast_enhancer_agent()`: Enhances forecasts with additional insights

2. **Task Definitions**:
   - `harvest_data()`: Gathers financial data from various sources
   - `enhance_forecast()`: Improves forecasts with additional analysis
   - `compose_report_part1()` and `compose_report_part2()`: Generate different sections of the analysis report

3. **Utility Functions**:
   - `get_appropriate_llm()`: Selects the appropriate language model based on task complexity
   - `harvest_data_offline()`: Collects data without requiring an active agent
   - `build_news_query()`: Constructs queries for retrieving relevant news

4. **Market Brief Generation**:
   - `build_market_brief()`: Main function that orchestrates the entire analysis process
   - `merge_news_into_results()`: Integrates news data with technical analysis

5. **Data Processing**:
   - Handles various timeframes (1-minute, 5-minute, hourly, daily, weekly)
   - Processes technical indicators and pattern recognition
   - Integrates company news with technical analysis

## Data Flow

1. User submits stock symbols through the web interface
2. At scheduled times, the application processes these symbols
3. For each symbol:
   - Financial data is fetched from Finnhub API
   - Technical analysis is performed
   - Reports are generated
   - Results are saved locally and optionally uploaded to S3

## Troubleshooting

- If the application fails to start, check that the Finnhub API token is correctly set in the `.env` file
- If analysis results are not being generated, check the logs for errors:
  ```bash
  docker-compose logs
  ```
- If S3 upload fails, verify AWS credentials and bucket permissions