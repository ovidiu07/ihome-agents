FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY research_stocks ./research_stocks
EXPOSE 8000
CMD ["uvicorn", "research_stocks.main:app", "--host", "0.0.0.0", "--port", "8000"]
