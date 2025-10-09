import os
import re
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple
from pathlib import Path
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
import faiss
import json
import uuid

logger = logging.getLogger(__name__)

# Глобальные переменные для моделей
EMBEDDING_MODEL = None
QA_MODEL = None
FAISS_INDEX = None
CONTEXTS_STORE = []


def initialize_models():
    """Инициализация моделей для QA системы"""
    global EMBEDDING_MODEL, QA_MODEL
    
    if EMBEDDING_MODEL is None:
        try:
            logger.info("Загрузка модели эмбеддингов...")
            EMBEDDING_MODEL = SentenceTransformer('intfloat/multilingual-e5-large')
            logger.info("✅ Модель эмбеддингов загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            # Fallback на более простую модель
            EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    if QA_MODEL is None:
        try:
            logger.info("Загрузка модели для ответов...")
            QA_MODEL = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Модель для ответов загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели для ответов: {e}")
            # Простая fallback модель
            QA_MODEL = None


def parse_docx_document(file_path: str) -> str:
    """Парсинг DOCX документа"""
    try:
        doc = DocxDocument(file_path)
        full_text = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text.strip())
        
        return '\n'.join(full_text)
    except Exception as e:
        logger.error(f"Ошибка парсинга DOCX: {e}")
        raise


def split_text_into_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Разбиение текста на чанки с перекрытием"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk + sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Добавляем перекрытие
                overlap_text = ' '.join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Генерация эмбеддингов для текстов"""
    model = initialize_models()
    if EMBEDDING_MODEL is None:
        raise RuntimeError("Модель эмбеддингов не инициализирована")
    
    try:
        embeddings = EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.error(f"Ошибка генерации эмбеддингов: {e}")
        raise


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Создание FAISS индекса для быстрого поиска"""
    global FAISS_INDEX
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    FAISS_INDEX = index
    return index


def search_similar_contexts(query: str, top_k: int = 10) -> List[Dict]:
    """Поиск похожих контекстов для запроса"""
    global FAISS_INDEX, CONTEXTS_STORE, EMBEDDING_MODEL
    
    if FAISS_INDEX is None or not CONTEXTS_STORE:
        raise RuntimeError("Индекс не инициализирован")
    
    # Генерация эмбеддинга для запроса
    query_embedding = EMBEDDING_MODEL.encode([query], convert_to_numpy=True)
    
    # Поиск в индексе
    distances, indices = FAISS_INDEX.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(CONTEXTS_STORE):
            result = {
                'context': CONTEXTS_STORE[idx],
                'score': float(1 / (1 + distance)),  # Конвертируем дистанцию в score
                'rank': i + 1
            }
            results.append(result)
    
    return results


def generate_answer(question: str, contexts: List[str]) -> str:
    """Генерация ответа на основе контекстов"""
    global QA_MODEL
    
    if QA_MODEL is None:
        # Простой fallback - возвращаем самый релевантный контекст
        if contexts:
            return f"На основе документа: {contexts[0][:500]}..."
        return "Извините, не могу найти ответ на ваш вопрос."
    
    # Формируем промпт
    context_text = "\n\n".join(contexts[:3])  # Берем топ-3 контекста
    prompt = f"""Контекст: {context_text}

Вопрос: {question}

Ответ:"""
    
    try:
        response = QA_MODEL(prompt, max_length=200, do_sample=True, temperature=0.7)
        answer = response[0]['generated_text'].split("Ответ:")[-1].strip()
        return answer
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        if contexts:
            return f"На основе документа: {contexts[0][:300]}..."
        return "Извините, произошла ошибка при генерации ответа."


def generate_qa_triplets(text: str, num_triplets: int = 5) -> List[Dict]:
    """Генерация триплетов контекст-вопрос-ответ для бенчмарка"""
    chunks = split_text_into_chunks(text, chunk_size=300)
    triplets = []
    
    for i, chunk in enumerate(chunks[:num_triplets]):
        # Простая генерация вопросов на основе контекста
        questions = generate_questions_from_context(chunk)
        
        for question in questions:
            answer = extract_answer_from_context(chunk, question)
            triplet = {
                'context': chunk,
                'question': question,
                'answer': answer,
                'chunk_index': i
            }
            triplets.append(triplet)
    
    return triplets


def generate_questions_from_context(context: str) -> List[str]:
    """Генерация вопросов на основе контекста"""
    # Простая эвристическая генерация вопросов
    questions = []
    
    # Ищем ключевые факты
    sentences = re.split(r'[.!?]+', context)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
            
        # Генерируем вопросы разных типов
        if any(word in sentence.lower() for word in ['является', 'составляет', 'равен']):
            # Вопрос "Что?"
            questions.append(f"Что {sentence.lower().split('является')[0].strip()}?")
        
        if any(word in sentence.lower() for word in ['где', 'находится', 'расположен']):
            # Вопрос "Где?"
            questions.append(f"Где {sentence.lower().split()[1:3]} расположен?")
        
        if any(word in sentence.lower() for word in ['когда', 'в году', 'дата']):
            # Вопрос "Когда?"
            questions.append(f"Когда произошло {sentence.lower().split()[1:3]}?")
    
    return questions[:3]  # Возвращаем максимум 3 вопроса


def extract_answer_from_context(context: str, question: str) -> str:
    """Извлечение ответа из контекста для вопроса"""
    # Простая эвристика - возвращаем наиболее релевантное предложение
    sentences = re.split(r'[.!?]+', context)
    
    question_words = set(question.lower().split())
    best_sentence = ""
    max_overlap = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        sentence_words = set(sentence.lower().split())
        overlap = len(question_words.intersection(sentence_words))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sentence = sentence
    
    return best_sentence if best_sentence else context[:200] + "..."


def calculate_metrics(predicted_answer: str, ground_truth: str) -> Dict[str, float]:
    """Вычисление метрик качества"""
    # Заглушки для метрик - в реальной системе нужно подключить соответствующие библиотеки
    metrics = {
        'bleurt_score': 0.0,  # Потребует установки BLEURT
        'semantic_similarity': 0.0,  # Потребует BGE-m3
        'rouge_l': 0.0,  # Потребует rouge-score
        'exact_match': 1.0 if predicted_answer.strip() == ground_truth.strip() else 0.0
    }
    
    # Простая метрика схожести слов
    pred_words = set(predicted_answer.lower().split())
    true_words = set(ground_truth.lower().split())
    
    if true_words:
        word_overlap = len(pred_words.intersection(true_words)) / len(true_words)
        metrics['word_overlap'] = word_overlap
    
    return metrics


def calculate_retrieval_metrics(retrieved_contexts: List[Dict], relevant_context: str) -> Dict[str, float]:
    """Вычисление метрик для ретривера"""
    # Простая реализация NDCG@10, MRR@10, MAP@100
    relevant_found = False
    rank = None
    
    for i, ctx in enumerate(retrieved_contexts[:10]):
        if ctx['context'] == relevant_context:
            relevant_found = True
            rank = i + 1
            break
    
    metrics = {
        'ndcg_10': 0.0,
        'mrr_10': 0.0,
        'map_100': 0.0
    }
    
    if relevant_found and rank:
        # Простые вычисления
        metrics['mrr_10'] = 1.0 / rank
        metrics['ndcg_10'] = 1.0 / np.log2(rank + 1)
        metrics['map_100'] = 1.0 / rank
    
    return metrics