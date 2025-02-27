FROM krmp-d2hub-idock.9rum.cc/goorm/python:3.6-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi torch uvicorn numpy langchain_openai langchain_community chromadb

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]