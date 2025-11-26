FROM python:3.12-slim

# Evitar buffering de logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Crear directorio de la app
WORKDIR /app

# Instalar dependencias del sistema (psycopg2, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalarlos
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el resto del código (por ahora la carpeta estará casi vacía)
COPY . /app/

# Puerto que usaremos en desarrollo
EXPOSE 8000

# Comando por defecto (se sobrescribe en docker-compose)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
