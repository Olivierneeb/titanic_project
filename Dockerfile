FROM python:3.12-slim

WORKDIR /app

# Ajouter le répertoire `src` au PYTHONPATH
ENV PYTHONPATH=/app/src

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
