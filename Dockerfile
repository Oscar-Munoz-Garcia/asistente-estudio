FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependencias primero: Docker reutiliza esta capa si requirements.txt no cambia
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Despues, el codigo
COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]