FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl \
    libffi-dev python3-dev libtool pkg-config \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && wget -O config.guess https://git.savannah.gnu.org/cgit/config.git/plain/config.guess \
    && wget -O config.sub https://git.savannah.gnu.org/cgit/config.git/plain/config.sub \
    && mv config.guess ta-lib/ && mv config.sub ta-lib/ \
    && cd ta-lib && ./configure --prefix=/usr && make && make install && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY research_stocks ./research_stocks
EXPOSE 8000