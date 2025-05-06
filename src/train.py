import mlflow
import mlflow.pytorch
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys
import os
import gc

# Настройка ограничений ресурсов
torch.set_num_threads(4)  # Ограничиваем количество потоков PyTorch
os.environ['OMP_NUM_THREADS'] = '4'  # Ограничиваем количество потоков OpenMP
os.environ['MKL_NUM_THREADS'] = '4'  # Ограничиваем количество потоков MKL

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    try:
        model_name = "cointegrated/rubert-tiny2"
        logger.info(f"Загрузка модели {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # Явно указываем использование CPU
        model = model.to('cpu')
        # Очищаем кэш CUDA если он есть
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Модель и токенизатор успешно загружены")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise

def encode_text(text, model, tokenizer):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Явно указываем использование CPU
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Очищаем неиспользуемую память
        del inputs
        gc.collect()
        return outputs.last_hidden_state.mean(dim=1).numpy()
    except Exception as e:
        logger.error(f"Ошибка при кодировании текста: {str(e)}")
        raise

def main():
    try:
        # Инициализация MLflow
        logger.info("Инициализация MLflow")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("question_search_engine")

        with mlflow.start_run():
            # Загрузка модели и токенизатора
            model, tokenizer = load_model_and_tokenizer()
            
            # Логирование параметров
            mlflow.log_param("model_name", "cointegrated/rubert-tiny2")
            mlflow.log_param("max_length", 512)
            mlflow.log_param("device", "cpu")
            mlflow.log_param("num_threads", torch.get_num_threads())

            # Пример использования модели
            questions = [
                "Как работает машинное обучение?",
                "Что такое глубокое обучение?",
                "Как использовать трансформеры?"
            ]

            logger.info("Получение эмбеддингов для вопросов")
            # Получение эмбеддингов
            embeddings = []
            for question in questions:
                embedding = encode_text(question, model, tokenizer)
                embeddings.append(embedding)
                # Очищаем неиспользуемую память после каждого вопроса
                gc.collect()

            # Сохранение модели
            logger.info("Сохранение модели в MLflow")
            mlflow.pytorch.log_model(model, "model")

            # Пример поиска похожих вопросов
            query = "Расскажи про нейронные сети"
            logger.info(f"Поиск похожих вопросов для запроса: {query}")
            query_embedding = encode_text(query, model, tokenizer)
            
            similarities = cosine_similarity(query_embedding, np.vstack(embeddings))[0]
            most_similar_idx = np.argmax(similarities)
            
            # Логирование метрик
            mlflow.log_metric("similarity_score", similarities[most_similar_idx])
            
            logger.info(f"Наиболее похожий вопрос: {questions[most_similar_idx]}")
            logger.info(f"Схожесть: {similarities[most_similar_idx]:.4f}")

            # Очищаем память перед завершением
            del model, tokenizer, embeddings, query_embedding
            gc.collect()

    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main() 