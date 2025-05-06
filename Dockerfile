FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание директории для MLflow
RUN mkdir -p /app/mlruns

# Копирование файлов проекта
COPY requirements.txt .
COPY src/ ./src/
COPY data/ ./data/

# Установка Python зависимостей с увеличенным таймаутом
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Открытие порта для MLflow UI
EXPOSE 5000

# Запуск MLflow сервера
CMD ["mlflow", "ui", "--host", "0.0.0.0"] 