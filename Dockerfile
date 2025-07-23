FROM python:3.11-slim
WORKDIR /app

# Install build tools + dependencies
RUN apt-get update && apt-get install -y \
    build-essential wget curl libffi-dev python3-dev libtool pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build and install the TA‑Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
 && tar xzf ta-lib-0.4.0-src.tar.gz \
 && cd ta-lib \
 && ./configure --prefix=/usr \
 && make \
 && make install \
 && ldconfig \
 && cd .. \
 && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set env for Python wrapper
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY research_stocks ./research_stocks
EXPOSE 8000
CMD ["uvicorn", "research_stocks.main:app", "--host", "0.0.0.0", "--port", "8000"]