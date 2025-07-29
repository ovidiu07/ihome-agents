FROM python:3.11-slim
WORKDIR /app

# Install system dependencies and TA-Lib C library
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements early
COPY requirements.txt .

# Install pip packages in two stages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel


# Set TA-Lib include and library paths for pip install
ENV TA_INCLUDE_PATH=/usr/include
ENV TA_LIBRARY_PATH=/usr/lib

# Then install the rest
RUN pip install --no-cache-dir -r requirements.txt

COPY research_stocks ./research_stocks
EXPOSE 8000