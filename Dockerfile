FROM python:3.11-slim
WORKDIR /app

# Install system dependencies and TA-Lib C library
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl \
    libffi-dev python3-dev libtool pkg-config \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && wget -O ta-lib/config.guess https://git.savannah.gnu.org/cgit/config.git/plain/config.guess \
    && wget -O ta-lib/config.sub https://git.savannah.gnu.org/cgit/config.git/plain/config.sub \
    && cd ta-lib && ./configure --prefix=/usr && make && make install && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements early
COPY requirements.txt .

# Install pip packages in two stages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel


# Set TA-Lib include and library paths for pip install
ENV TA_INCLUDE_PATH=/usr/include
ENV TA_LIBRARY_PATH=/usr/lib

RUN test -f /usr/lib/libta_lib.so && echo "TA-Lib C library is in place ✅"

RUN ln -s /usr/lib/libta_lib.so /usr/lib/libta-lib.so && \
    pip install --no-cache-dir TA-Lib==0.4.0

# Build TA-Lib Python bindings manually
#RUN git clone https://github.com/mrjbq7/ta-lib.git && \
#    cd ta-lib && python3 setup.py build && python3 setup.py install && \
#    cd .. && rm -rf ta-lib \

# Then install the rest
RUN pip install --no-cache-dir -r requirements.txt

COPY research_stocks ./research_stocks
EXPOSE 8000