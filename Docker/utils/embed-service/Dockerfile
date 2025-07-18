FROM python:3.12-slim

RUN python -m pip install --upgrade pip && \
    pip install fastapi sentence-transformers uvicorn

RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models && \
    cd /app/models && \
    wget --no-check-certificate https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v1.zip

RUN mkdir -p /app/models/all-MiniLM-L6 && \
    unzip /app/models/all-MiniLM-L6-v1.zip -d /app/models/all-MiniLM-L6

WORKDIR /app

COPY ./main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]