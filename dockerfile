FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py serve.py evaluate.py /app/

# Train model during build
RUN python train.py

EXPOSE 8000

CMD ["python", "serve.py"]
