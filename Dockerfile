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
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.getenv('PORT', '8000'))" || exit 1

# El puerto lo da la variable PORT (la nube la define); en local usa el 8000.
# "exec" hace que la API reciba las senales de parada y se apague limpiamente.
CMD ["sh", "-c", "exec fastapi run app/main.py --port ${PORT:-8000}"]