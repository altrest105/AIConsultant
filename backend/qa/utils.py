import os
import logging
from pathlib import Path

new_cache_dir = Path("D:/Programs/HuggingFace_Cache") 
new_cache_dir.mkdir(parents=True, exist_ok=True) # Создаем папку, если ее нет

# Установка переменной окружения HF_HOME
# HuggingFace будет использовать эту папку для всего кэша (модели, датасеты, токены)
os.environ['HF_HOME'] = str(new_cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(new_cache_dir / "models")

from pydantic import Field, PrivateAttr
from typing import List, Dict, Any, Optional
import torch
import pickle
import json
import re
import time
from tqdm import tqdm
import random
import numpy as np
import warnings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline,
    set_seed
)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from sentence_transformers import CrossEncoder

from rank_bm25 import BM25Okapi

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QA_SYSTEM")

# Игнорирование предупреждений BitsAndBytes
warnings.filterwarnings("ignore", category=UserWarning)

# --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ПЕРЕМЕННЫЕ ---
VECTOR_STORE: Optional[Qdrant] = None
EMBEDDINGS_MODEL: Optional[HuggingFaceEmbeddings] = None
RERANKER: Optional[CrossEncoder] = None
LLM_MODEL: Optional[HuggingFacePipeline] = None
BM25_INDEX: Optional[BM25Okapi] = None
BM25_CORPUS: List[str] = []

# --- КОНФИГУРАЦИЯ СИСТЕМЫ ---
CONFIG = {
    "mode": "balanced",
    
    # Гиперпараметры поиска
    "K_VEC": 20,          # Количество векторов из Qdrant
    "K_BM25": 20,         # Количество текстов из BM25
    "ALPHA": 0.5,         # Коэффициент RRF (Reciprocal Rank Fusion)
    "TOP_N_RERANKER": 5,  # Финальный контекст для LLM
    "USE_HYDE": True,
    "SEED": 42,           # Random seed для воспроизводимости
    
    # Параметры LLM
    "LLM_MODEL_NAME": "Qwen/Qwen2.5-1.5B-Instruct",
    "EMBEDDING_MODEL_NAME": "BAAI/bge-m3",
    "RERANKER_MODEL_NAME": "BAAI/bge-reranker-v2-m3",
    
    # Конфигурация сплиттинга документов
    "CHUNK_SIZE": 400,
    "CHUNK_OVERLAP": 50,
    
    # Конфигурация бенчмарка
    "BENCHMARK": {
        "N_QUESTIONS": 100,
        "types": {
            "factual": 0.25,
            "analytical": 0.25,
            "procedural": 0.20,
            "comparative": 0.15,
            "synthetic": 0.15
        },
        "difficulty_levels": {
            "easy": {
                "description": "Простой факт, находится в одном предложении",
                "answer_length_range": [20, 100],
                "chunks_required": 1
            },
            "medium": {
                "description": "Требует обобщения информации из 2-3 предложений",
                "answer_length_range": [100, 250],
                "chunks_required": 1
            },
            "hard": {
                "description": "Требует синтеза из 2-х разных секций документа",
                "answer_length_range": [250, 400],
                "chunks_required": 2
            },
            "expert": {
                "description": "Требует глубокого понимания, сравнения, вывода неявных связей",
                "answer_length_range": [400, 600],
                "chunks_required": 3
            }
        }
    }
}

# --- УТИЛИТЫ ---

def set_deterministic_mode(seed: int = CONFIG["SEED"]):
    """
    Устанавливает глобальный seed для всех библиотек (Python, NumPy, PyTorch, Transformers).
    
    Это обеспечивает полную воспроизводимость результатов RAG-системы.
    
    Args:
        seed: Числовой seed для установки.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)
    logger.info(f"⚙️  Система установлена в детерминированный режим с SEED: {seed}")

def _initialize_llm():
    """
    Инициализирует Large Language Model (Qwen2.5-7B) с 4-битной квантизацией.
    
    Загружает модель, токенизатор и настраивает HuggingFacePipeline для
    детерминированной (do_sample=False) генерации. Результат сохраняется
    в глобальной переменной LLM_MODEL.
    """
    global LLM_MODEL
    logger.info(f"⏳ Загрузка LLM: {CONFIG['LLM_MODEL_NAME']} (4-bit, {CONFIG['mode']})")
    
    try:
        # 4-bit quantization config (NF4)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["LLM_MODEL_NAME"])
        
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["LLM_MODEL_NAME"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Настройка pipeline для детерминированной генерации
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            return_full_text=False
        )
        LLM_MODEL = HuggingFacePipeline(pipeline=pipe)
        logger.info("✅ LLM загружена и настроена.")
    
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке LLM: {e}")
        LLM_MODEL = None

def _initialize_embeddings():
    """
    Инициализирует модель эмбеддингов (BGE-M3) с использованием HuggingFaceEmbeddings.
    
    Результат сохраняется в глобальной переменной EMBEDDINGS_MODEL.
    """
    global EMBEDDINGS_MODEL
    logger.info(f"⏳ Загрузка Embedding: {CONFIG['EMBEDDING_MODEL_NAME']}")
    try:
        EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name=CONFIG["EMBEDDING_MODEL_NAME"],
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        logger.info("✅ Embedding модель загружена.")
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке Embedding: {e}")
        EMBEDDINGS_MODEL = None

def _initialize_reranker():
    """
    Инициализирует Cross-Encoder Reranker (BGE-Reranker-v2-m3).
    
    Используется для повышения точности релевантности после гибридного поиска.
    Результат сохраняется в глобальной переменной RERANKER.
    """
    global RERANKER
    logger.info(f"⏳ Загрузка Reranker: {CONFIG['RERANKER_MODEL_NAME']}")
    try:
        RERANKER = CrossEncoder(
            CONFIG["RERANKER_MODEL_NAME"], 
            max_length=512,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("✅ Reranker загружен.")
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке Reranker: {e}")
        RERANKER = None

def _initialize_vector_store():
    """
    Инициализирует Qdrant клиент и векторное хранилище.
    
    Создает локальную коллекцию "transconsultant_kb", если она отсутствует.
    Результат сохраняется в глобальной переменной VECTOR_STORE.
    """
    global VECTOR_STORE
    logger.info("⏳ Инициализация Qdrant...")
    try:
        client = QdrantClient(path="./qdrant_storage") 
        
        embedding_size = 1024 # BGE-M3 dimensionality
        
        # Проверка и создание коллекции
        collection_name = "transconsultant_kb"
        try:
            client.get_collection(collection_name)
            logger.info(f"✅ Qdrant: Коллекция '{collection_name}' существует.")
        except Exception as e:
             client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
            )
             logger.info(f"✅ Qdrant: Коллекция '{collection_name}' создана.")
             
        # Создание объекта LangChain VectorStore
        VECTOR_STORE = Qdrant(
            client=client, 
            collection_name=collection_name, 
            embeddings=EMBEDDINGS_MODEL
        )
        logger.info("✅ Qdrant инициализирован.")
    
    except Exception as e:
        logger.error(f"❌ Ошибка при инициализации Qdrant: {e}")
        VECTOR_STORE = None

def initialize_qa_system(mode: str = CONFIG["mode"], force_reload: bool = False):
    """
    Инициализирует все компоненты QA-системы (LLM, Embeddings, Reranker, Qdrant).
    
    Args:
        mode: Режим работы системы (по умолчанию 'balanced'). Не используется в данной версии.
        force_reload: Флаг для принудительной перезагрузки всех моделей и хранилищ.
    """
    global LLM_MODEL, EMBEDDINGS_MODEL, RERANKER, VECTOR_STORE
    
    set_deterministic_mode()
    
    if force_reload or not (LLM_MODEL and EMBEDDINGS_MODEL and RERANKER and VECTOR_STORE):
        logger.info("\n" + "="*60)
        logger.info(f"🚀 ИНИЦИАЛИЗАЦИЯ QA-СИСТЕМЫ ({mode})")
        logger.info("="*60)
        
        if not EMBEDDINGS_MODEL:
            _initialize_embeddings()
        
        if not RERANKER:
            _initialize_reranker()
            
        if EMBEDDINGS_MODEL and not VECTOR_STORE:
            _initialize_vector_store()
            
        if not LLM_MODEL:
            _initialize_llm()
            
        logger.info("✅ ВСЕ КОМПОНЕНТЫ ЗАГРУЖЕНЫ.")
    else:
        logger.info("⚙️  Система уже инициализирована. Используется текущая конфигурация.")

# --- ПАРСИНГ И ИНДЕКСАЦИЯ ---

def _parse_document_universal(file_path: Path) -> List[Document]:
    """
    Парсит DOCX и TXT, разделяет на чанки
    и возвращает список LangChain Document.
    
    Используется для сохранения структуры документа (DOCX, TXT).
    
    Args:
        file_path: Путь к файлу.
        
    Returns:
        Список объектов Document с метаданными.
    """
    logger.info(f"📄 Парсинг файла: {file_path.name}")
    
    full_text = ""
    file_type = file_path.suffix.lstrip('.')
    
    try:
        if file_type == 'docx':
            # Используем python-docx
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            # Извлекаем текст из всех параграфов
            full_text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        else:
            logger.warning(f"⚠️ Неподдерживаемый формат файла: .{file_type}")
            return []
            
        if not full_text.strip():
             logger.warning(f"⚠️ Файл пуст или содержит только пробелы: {file_path.name}")
             return []

        logger.info(f"✅ Файл распарсен: {len(full_text)} символов")
        
        # Рекурсивный сплиттер для создания чанков
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"],
            separators=["\n\n\n", "\n\n", "\n", " ", ""] # Оставляем разделители для структуры
        )
        
        documents = text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{"source": file_path.name, "doc_type": file_type}]
        )
        
        logger.info(f"📚 Разделено на {len(documents)} чанков.")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Ошибка при парсинге {file_path.name}: {e}")
        return []

def update_bm25_index(documents: List[Document]):
    """
    Обновляет BM25 индекс и корпус, сохраняя корпус для будущих сессий.
    
    Args:
        documents: Список LangChain Document для добавления в индекс.
    """
    global BM25_INDEX, BM25_CORPUS
    
    # Извлечение текста
    texts = [doc.page_content for doc in documents]
    
    # Токенизация для BM25
    tokenized_corpus = [doc.split(" ") for doc in texts]
    
    # Обновление корпуса и индекса
    BM25_CORPUS.extend(texts)
    BM25_INDEX = BM25Okapi(tokenized_corpus)
    
    # Сохранение корпуса
    with open("bm25_corpus.pkl", "wb") as f:
        pickle.dump(BM25_CORPUS, f)
    
    logger.info(f"✅ BM25 индекс обновлен: {len(BM25_CORPUS)} документов")

def index_document(file_path: Path):
    """
    Выполняет полный цикл индексации для одного документа: парсинг,
    добавление в векторную БД (Qdrant) и обновление BM25 индекса.
    
    Args:
        file_path: Путь к файлу.
    """
    global VECTOR_STORE
    
    documents = _parse_document_universal(file_path)
    
    if not documents:
        return

    # 1. Векторная БД (Qdrant)
    if VECTOR_STORE:
        try:
            VECTOR_STORE.add_documents(documents)
            logger.info(f"✅ Qdrant: Добавлено {len(documents)} чанков.")
        except Exception as e:
            logger.error(f"❌ Ошибка Qdrant: {e}")

    # 2. BM25 (Лексический поиск)
    update_bm25_index(documents)

def index_knowledge_base(folder_path: Path):
    """
    Индексирует все поддерживаемые документы (.docx, .txt) из указанной папки.
    
    Args:
        folder_path: Путь к папке с исходными документами.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"📚 ИНДЕКСАЦИЯ БАЗЫ ЗНАНИЙ: {folder_path}")
    logger.info(f"{'='*60}\n")
    
    if not folder_path.exists():
        logger.error(f"❌ Папка не найдена: {folder_path}")
        return
    
    supported_formats = ['.docx', '.txt']
    
    files_to_index = []
    for ext in supported_formats:
        files_to_index.extend(folder_path.glob(f"*{ext}"))
    
    if not files_to_index:
        logger.warning(f"⚠️ Нет файлов для индексации в {folder_path}")
        return
    
    logger.info(f"📄 Найдено файлов: {len(files_to_index)}")
    
    for file_path in tqdm(files_to_index, desc="Индексация"):
        index_document(file_path)
        
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    logger.info(f"   Всего документов в корпусе: {len(BM25_CORPUS)}")
    logger.info(f"{'='*60}\n")

# --- РЕТРИВЕР И RAG ---

class EnhancedHybridRetriever(BaseRetriever):
    """
    Кастомный ретривер, реализующий гибридный поиск (Vector + BM25) 
    с использованием Reciprocal Rank Fusion (RRF) и финальным Reranker.
    """
    
    # Объявление основных компонентов как полей Pydantic
    vector_store: Qdrant = Field(..., description="LangChain Qdrant VectorStore instance.")
    reranker: CrossEncoder = Field(..., description="Sentence Transformer CrossEncoder instance.")

    # Объявление остальных атрибутов (если они должны быть установлены при инициализации)
    # Используем PrivateAttr или просто атрибуты класса/экземпляра, в зависимости от требований.
    k_vec: int = CONFIG["K_VEC"]
    k_bm25: int = CONFIG["K_BM25"]
    alpha: float = CONFIG["ALPHA"]
    top_n: int = CONFIG["TOP_N_RERANKER"]

    # Добавляем Config для разрешения нестандартных типов данных (Qdrant, CrossEncoder)
    class Config:
        arbitrary_types_allowed = True
        
    # Удалить метод __init__, т.к. он теперь обрабатывается Pydantic. 
    # Если нужна дополнительная инициализация, использовать метод _post_init_ или validate

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Выполняет поиск релевантных документов.

        Args:
            query: Исходный запрос или расширенный запрос (HyDE).
            **kwargs: Дополнительные аргументы (игнорируются).

        Returns:
            Список наиболее релевантных объектов Document (после RRF и Reranking).
        """
        global BM25_INDEX, BM25_CORPUS
        
        # 1. Vector Search
        vec_results = self.vector_store.similarity_search_with_score(query, k=self.k_vec)
        
        # 2. BM25 Search
        bm25_results = []
        if BM25_INDEX and BM25_CORPUS:
            tokenized_query = query.split(" ")
            doc_scores = BM25_INDEX.get_scores(tokenized_query)
            
            # Get indices of top-K documents
            top_n_indices = np.argsort(doc_scores)[::-1][:self.k_bm25]
            
            for rank, idx in enumerate(top_n_indices):
                # Create Document for RRF fusion
                doc = Document(
                    page_content=BM25_CORPUS[idx],
                    metadata={"source": "bm25_hit", "rank": rank + 1}
                )
                bm25_results.append(doc)
        
        # 3. RRF (Reciprocal Rank Fusion)
        all_docs_map: Dict[str, Dict[str, Any]] = {}

        # RRF score calculation
        def calculate_rrf_score(rank, k=60):
            return 1 / (k + rank)

        # Process vector results
        for rank, (doc, score) in enumerate(vec_results):
            text_hash = hash(doc.page_content)
            
            if text_hash not in all_docs_map:
                all_docs_map[text_hash] = {"doc": doc, "rrf_score": 0.0}
            
            rrf_score = calculate_rrf_score(rank + 1)
            all_docs_map[text_hash]["rrf_score"] += rrf_score * self.alpha

        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            text_hash = hash(doc.page_content)
            
            if text_hash not in all_docs_map:
                all_docs_map[text_hash] = {"doc": doc, "rrf_score": 0.0}
            
            rrf_score = calculate_rrf_score(rank + 1)
            all_docs_map[text_hash]["rrf_score"] += rrf_score * (1.0 - self.alpha)

        # Sort by RRF Score
        final_candidates = sorted(
            all_docs_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Extract documents for reranking
        rerank_input = [item["doc"] for item in final_candidates]
        
        # 4. Reranking
        if self.reranker and len(rerank_input) > 0:
            pairs = [[query, doc.page_content] for doc in rerank_input]
            scores = self.reranker.predict(pairs)
            
            # Merge documents and scores
            scored_documents = sorted(
                zip(rerank_input, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # 5. Return top-N documents
            final_docs = [doc for doc, score in scored_documents[:self.top_n]]
        else:
            # Fallback to top-N based on RRF
            final_docs = [item["doc"] for item in final_candidates[:self.top_n]]

        return final_docs

def _apply_hyde(question: str) -> str:
    """
    Генерирует гипотетический ответ (HyDE) для расширения исходного запроса
    и повышения качества поиска.

    Args:
        question: Исходный вопрос пользователя.

    Returns:
        Расширенный запрос, объединяющий исходный вопрос и гипотетический ответ.
    """
    global LLM_MODEL
    if not LLM_MODEL:
        logger.warning("⚠️ LLM не инициализирована для HyDE.")
        return question

    # HyDE prompt
    hyde_prompt = (
        "Ты — помощник, который должен дать краткий, но полный гипотетический ответ "
        "на вопрос, чтобы улучшить поиск релевантных документов. Отвечай на основе своих внутренних знаний. "
        f"Вопрос: {question}"
    )

    try:
        response = LLM_MODEL.pipeline(
            hyde_prompt,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0,
            return_full_text=False
        )
        
        hypothetical_answer = response[0]['generated_text'].strip()
        expanded_query = f"{question} {hypothetical_answer}"
        
        logger.debug(f"📝 HyDE: '{hypothetical_answer[:80]}...'")
        return expanded_query
    
    except Exception as e:
        logger.warning(f"⚠️  Ошибка HyDE: {e}")
        return question

def answer_question(question: str) -> Dict[str, Any]:
    """
    Основная функция для ответа на вопрос с использованием RAG-цепочки.
    
    Применяет HyDE (если включен), выполняет гибридный поиск и генерирует
    ответ с помощью LLM с инструкцией Chain-of-Thought (CoT).

    Args:
        question: Вопрос пользователя.

    Returns:
        Словарь с ответом, найденными исходными документами и задержкой:
            - answer (str): Финальный ответ LLM.
            - source_documents (List[Document]): Список источников.
            - latency_sec (float): Время выполнения запроса в секундах.
    """
    global LLM_MODEL, VECTOR_STORE, RERANKER
    
    if not (LLM_MODEL and VECTOR_STORE and RERANKER):
        logger.error("❌ Система не инициализирована.")
        return {"answer": "Система QA не инициализирована.", "source_documents": []}
    
    start_time = time.time()
    
    # 1. HyDE application
    expanded_query = _apply_hyde(question) if CONFIG["USE_HYDE"] else question

    # 2. Retriever setup
    retriever = EnhancedHybridRetriever(vector_store=VECTOR_STORE, reranker=RERANKER)
    
    # 3. RetrievalQA Chain setup
    QA_PROMPT = PromptTemplate(
        template=(
            "ТЫ — ФАКТОЛОГИЧЕСКИЙ БОТ. ОТВЕЧАЙ СТРОГО, ИСПОЛЬЗУЯ ТОЛЬКО КОНТЕКСТ. "
            "Сначала проведи АНАЛИЗ (цепочку рассуждений) того, как контекст отвечает на вопрос. "
            "Затем дай КРАТКИЙ, но полный ответ, используя строгий формат.\n\n"
            "КРИТИЧЕСКИЕ ПРАВИЛА: "
            "1. Не говори, что контекст не найден, если ответ можно составить. "
            "2. Ответ должен быть только на русском языке. "
            "3. Если контекст НЕ содержит ответа, скажи 'Недостаточно информации для ответа.'\n\n"
            "КОНТЕКСТ:\n{context}\n\n"
            "ВОПРОС: {question}\n\n"
            "АНАЛИЗ: " # Chain-of-Thought instruction
        ),
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=LLM_MODEL,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": expanded_query}) 
    
    end_time = time.time()
    latency = end_time - start_time
    
    # 4. Final answer extraction
    raw_answer = result['result'].strip()
    match = re.search(r'ОТВЕТ:\s*(.*)', raw_answer, re.DOTALL | re.IGNORECASE) 
    
    if match:
        final_answer = match.group(1).strip()
    else:
        # Fallback: use entire text if "ОТВЕТ:" label is missing
        final_answer = raw_answer
    
    logger.info(f"⏱️  Задержка: {latency:.2f}s")
    
    return {
        "answer": final_answer,
        "source_documents": result["source_documents"],
        "latency_sec": latency
    }


# --- ФУНКЦИИ БЕНЧМАРКА ---

def generate_diverse_benchmark(corpus: List[str], N: int = 100) -> List[Dict[str, Any]]:
    """
    Генерирует набор вопросов (бенчмарк), распределенных по типу и сложности,
    используя LLM для создания вопросов на основе предоставленного корпуса.
    
    Args:
        corpus: Список текстовых чанков, на основе которых будут генерироваться вопросы.
        N: Общее количество вопросов для генерации.

    Returns:
        Список словарей, где каждый словарь представляет собой вопрос
        с метаданными ('question', 'type', 'difficulty', 'min_chunks_required').
    """
    global LLM_MODEL
    if not LLM_MODEL or not corpus:
        logger.error("❌ LLM или Корпус не инициализированы для бенчмарка.")
        return []

    type_weights = CONFIG["BENCHMARK"]["types"]
    difficulty_map = CONFIG["BENCHMARK"]["difficulty_levels"]
    
    # Выбор случайных документов (часть корпуса) для генерации
    num_docs_for_gen = min(10, len(corpus))
    selected_docs = random.sample(corpus, num_docs_for_gen)
    context_for_generation = "\n\n---\n\n".join(selected_docs)

    benchmark_questions = []

    for q_type, weight in type_weights.items():
        num_q_for_type = round(N * weight)
        if num_q_for_type == 0: continue

        # Случайный выбор сложности для данного типа
        difficulty_name = random.choice(list(difficulty_map.keys()))
        difficulty_config = difficulty_map[difficulty_name]
        
        generation_prompt = (
            f"Используя следующий КОНТЕКСТ, сгенерируй {num_q_for_type} вопросов "
            f"типа '{q_type}' и сложности '{difficulty_name}'. "
            f"Требуемая сложность: {difficulty_config['description']}. "
            f"Вопросы должны быть уникальными и требовать {difficulty_config['chunks_required']} чанков для ответа.\n\n"
            f"КОНТЕКСТ: {context_for_generation}\n\n"
            "ГЕНЕРИРУЙ ТОЛЬКО СПИСОК ВОПРОСОВ (по одному на строку) БЕЗ НУМЕРАЦИИ И ЛЮБОГО ДРУГОГО ТЕКСТА."
        )

        try:
            # Детерминированная генерация вопросов
            response = LLM_MODEL.pipeline(
                generation_prompt,
                max_new_tokens=4096,
                do_sample=False, 
                temperature=0.0,
                return_full_text=False
            )
            
            generated_text = response[0]['generated_text'].strip()
            # Фильтрация и очистка вопросов
            questions = [
                q.strip() for q in generated_text.split('\n') 
                if q.strip() and len(q.strip()) > 10
            ]
            
            for q in questions:
                benchmark_questions.append({
                    "question": q,
                    "type": q_type,
                    "difficulty": difficulty_name,
                    "min_chunks_required": difficulty_config['chunks_required']
                })

            logger.info(f"   ✅ Сгенерировано {len(questions)} вопросов типа '{q_type}'")

        except Exception as e:
            logger.error(f"❌ Ошибка при генерации вопросов типа {q_type}: {e}")
            continue

    logger.info(f"📊 Всего сгенерировано вопросов: {len(benchmark_questions)}")
    return benchmark_questions

def run_benchmark(benchmark_questions: List[Dict[str, Any]], save_path: str = "benchmark_results.json") -> Dict[str, Any]:
    """
    Запускает оценку QA-системы на наборе вопросов и сохраняет результаты в JSON.

    Args:
        benchmark_questions: Список вопросов, сгенерированных функцией generate_diverse_benchmark.
        save_path: Путь для сохранения JSON файла с результатами.

    Returns:
        Словарь с общими метриками и детальными результатами выполнения каждого запроса.
    """
    logger.info(f"\n{'='*60}")
    logger.info("⚡️ ЗАПУСК БЕНЧМАРКА (ДЕГЕРМИНИРОВАННЫЙ ТЕСТ)")
    logger.info("="*60 + "\n")
    
    results = []
    successful_answers = 0
    total_latency = 0.0
    
    for item in tqdm(benchmark_questions, desc="Оценка RAG"):
        question = item["question"]
        
        try:
            result = answer_question(question)
            
            results.append({
                "question": question,
                "type": item["type"],
                "difficulty": item["difficulty"],
                "answer": result["answer"],
                "latency_sec": result["latency_sec"],
                "sources_found": [doc.metadata.get("source", "N/A") for doc in result["source_documents"]]
            })
            
            total_latency += result["latency_sec"]
            if result["answer"] not in ["Система QA не инициализирована.", "Недостаточно информации для ответа."]:
                successful_answers += 1
                
        except Exception as e:
            results.append({
                "question": question,
                "type": item["type"],
                "difficulty": item["difficulty"],
                "answer": f"ERROR: {e}",
                "latency_sec": 0.0,
                "sources_found": []
            })
            
    # Подсчет метрик
    total_questions = len(benchmark_questions)
    final_metrics = {
        "total_questions": total_questions,
        "successful_answers": successful_answers,
        "success_rate": successful_answers / total_questions if total_questions > 0 else 0.0,
        "avg_latency_sec": total_latency / successful_answers if successful_answers > 0 else 0.0,
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ БЕНЧМАРК ЗАВЕРШЕН")
    logger.info(f"📊 Успешных ответов: {final_metrics['successful_answers']}/{final_metrics['total_questions']} ({(final_metrics['success_rate']*100):.2f}%)")
    logger.info(f"⏱️  Средняя задержка: {final_metrics['avg_latency_sec']:.2f}s")
    logger.info(f"{'='*60}")
    
    # Сохранение
    output = {
        "results": results,
        "metrics": final_metrics
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Результаты сохранены: {save_path}")
    
    return output


# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---

# --- ОБНОВЛЕННЫЙ MAIN С ЗАЩИТОЙ ОТ ОШИБОК ---


if __name__ == '__main__':
    try:
        logger.info("="*60)
        logger.info("🚀 ЗАПУСК QA-СИСТЕМЫ TRANSCONSULTANT")
        logger.info("="*60 + "\n")
        
        # 1. КРИТИЧНО: Инициализация системы
        logger.info("Шаг 1/4: Инициализация компонентов...")
        initialize_qa_system()
        
        # 2. Проверка базы знаний
        logger.info("\nШаг 2/4: Проверка базы знаний...")
        docs_folder_path = Path("docs")
        
        if not docs_folder_path.exists():
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Папка не найдена: {docs_folder_path.resolve()}")
            logger.info("💡 РЕШЕНИЕ: Создайте папку 'docs' в корне проекта:")
            logger.info(f"   mkdir {docs_folder_path.resolve()}")
            logger.info("   Поместите туда файлы .docx, .txt")
            exit(1)
        
        # Проверка наличия файлов
        supported_formats = ['.docx', '.txt']
        all_files = []
        for ext in supported_formats:
            all_files.extend(list(docs_folder_path.glob(f"*{ext}")))
        
        if not all_files:
            logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Нет файлов для индексации в {docs_folder_path.resolve()}")
            logger.info("💡 РЕШЕНИЕ: Добавьте файлы .docx или .txt в папку docs/")
            exit(1)
        
        logger.info(f"✅ Найдено {len(all_files)} файлов для индексации")
        
        # 3. Индексация
        logger.info("\nШаг 3/4: Индексация базы знаний...")
        index_knowledge_base(docs_folder_path)
        
        if not BM25_CORPUS:
            logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Корпус пуст после индексации")
            exit(1)
        
        logger.info(f"✅ Корпус готов: {len(BM25_CORPUS)} чанков")
        
        # 4. Генерация бенчмарка
        logger.info("\nШаг 4/4: Генерация бенчмарка...")
        
        if len(BM25_CORPUS) < 5:
            logger.warning(f"⚠️ Корпус слишком мал ({len(BM25_CORPUS)} чанков). Минимум: 5")
            logger.info("💡 Добавьте больше документов в папку docs/")
            exit(1)
        
        benchmark = generate_diverse_benchmark(BM25_CORPUS, N=CONFIG["BENCHMARK"]["N_QUESTIONS"])
        
        if not benchmark:
            logger.error("❌ Не удалось сгенерировать бенчмарк")
            exit(1)
        
        # Сохранение вопросов
        with open("benchmark_questions.json", "w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Бенчмарк сохранен: benchmark_questions.json ({len(benchmark)} вопросов)")
        
        # 5. Запуск оценки
        logger.info("\n" + "="*60)
        logger.info("🎯 ЗАПУСК ОЦЕНКИ БЕНЧМАРКА")
        logger.info("="*60 + "\n")
        
        run_benchmark(benchmark, save_path="final_benchmark_metrics.json")
        
        logger.info("\n" + "="*60)
        logger.info("✅ ВСЕ ОПЕРАЦИИ УСПЕШНО ЗАВЕРШЕНЫ")
        logger.info("="*60)
        logger.info("\nФайлы результатов:")
        logger.info("  • benchmark_questions.json - Сгенерированные вопросы")
        logger.info("  • final_benchmark_metrics.json - Метрики оценки")
        logger.info("  • bm25_corpus.pkl - Индекс BM25")
        logger.info("  • qdrant_storage/ - Векторная БД")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Прервано пользователем (Ctrl+C)")
        exit(0)
        
    except Exception as e:
        logger.error("\n" + "="*60)
        logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА")
        logger.error("="*60)
        logger.error(f"Тип ошибки: {type(e).__name__}")
        logger.error(f"Сообщение: {str(e)}")
        logger.error("\nСтек вызовов:", exc_info=True)
        logger.error("\n💡 ВОЗМОЖНЫЕ ПРИЧИНЫ:")
        logger.error("  1. Недостаточно VRAM (требуется ~8GB)")
        logger.error("  2. Проблемы с CUDA (проверьте: nvidia-smi)")
        logger.error("  3. Битые файлы в папке docs/")
        logger.error("  4. Отсутствие зависимостей (pip install -r requirements.txt)")
        exit(1)



# if __name__ == '__main__':
#     try:
#         logger.info("="*60)
#         logger.info("🚀 ЗАПУСК QA-СИСТЕМЫ TRANSCONSULTANT")
#         logger.info("="*60 + "\n")
        
#         # 1. КРИТИЧНО: Инициализация системы
#         logger.info("Шаг 1/3: Инициализация компонентов...")
#         initialize_qa_system()
        
#         # 2. Проверка базы знаний
#         logger.info("\nШаг 2/3: Проверка базы знаний...")
#         docs_folder_path = Path("docs")
        
#         if not docs_folder_path.exists():
#             logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Папка не найдена: {docs_folder_path.resolve()}")
#             logger.info("💡 РЕШЕНИЕ: Создайте папку 'docs' и поместите туда файлы .docx, .txt")
#             exit(1)
        
#         # Проверка наличия файлов
#         supported_formats = ['.docx', '.txt']
#         all_files = []
#         for ext in supported_formats:
#             all_files.extend(list(docs_folder_path.glob(f"*{ext}")))
        
#         if not all_files:
#             logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Нет файлов для индексации в {docs_folder_path.resolve()}")
#             logger.info("💡 РЕШЕНИЕ: Добавьте файлы .docx или .txt в папку docs/")
#             exit(1)
        
#         logger.info(f"✅ Найдено {len(all_files)} файлов для индексации")
        
#         # 3. Индексация
#         logger.info("\nШаг 3/3: Индексация базы знаний...")
#         index_knowledge_base(docs_folder_path)
        
#         if not BM25_CORPUS:
#             logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Корпус пуст после индексации")
#             exit(1)
        
#         logger.info(f"✅ Корпус готов: {len(BM25_CORPUS)} чанков")
        
#         # 4. 🔥 ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ answer_question
#         logger.info("\n" + "="*60)
#         logger.info("✨ СИСТЕМА ГОТОВА К ТЕСТИРОВАНИЮ answer_question()")
#         logger.info("="*60)
        
#         # Выполняем тестовый запрос из примера
#         test_question = "Какова длина нефтепровода Нижневартовск — Усть-Балык?"
#         logger.info(f"\n💡 ТЕСТ: Выполняем тестовый запрос: '{test_question}'")
        
#         result = answer_question(test_question)
        
#         print("\n" + "="*60)
#         print("🔍 РЕЗУЛЬТАТ ТЕСТОВОГО ЗАПРОСА:")
#         print(f"  ВОПРОС: {test_question}")
#         print(f"  ОТВЕТ: {result['answer']}")
#         print(f"  ЗАДЕРЖКА: {result['latency_sec']:.2f} сек")
#         print(f"  ИСТОЧНИКИ: {result['source_documents']}")
#         print("="*60)
        
#         # Бесконечный цикл для ручного ввода
#         logger.info("💡 СИСТЕМА ГОТОВА. Введите 'exit' или 'quit' для завершения.")
#         while True:
#             user_input = input("❓ Ваш вопрос (или 'exit'): ")
#             if user_input.lower() in ['exit', 'quit']:
#                 break
            
#             if user_input.strip():
#                 test_result = answer_question(user_input)
#                 print("\n" + "="*60)
#                 print(f"✅ ОТВЕТ: {test_result['answer']}")
#                 print(f"⏱️  Задержка: {test_result['latency_sec']:.2f}s")
#                 print(f"📚 Источники: {test_result['source_documents']}")
#                 print("="*60 + "\n")


#     except KeyboardInterrupt:
#         logger.warning("\n⚠️ Прервано пользователем (Ctrl+C)")
#         exit(0)
        
#     except Exception as e:
#         logger.error("\n" + "="*60)
#         logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА")
#         logger.error("="*60)
#         logger.error(f"Тип ошибки: {type(e).__name__}")
#         logger.error(f"Сообщение: {str(e)}")
#         logger.error("\nСтек вызовов:", exc_info=True)
#         logger.error("\n💡 ВОЗМОЖНЫЕ ПРИЧИНЫ:")
#         logger.error("  1. Недостаточно VRAM (требуется ~8GB)")
#         logger.error("  2. Проблемы с CUDA (проверьте: nvidia-smi)")
#         logger.error("  3. Битые файлы в папке docs/")
#         logger.error("  4. Отсутствие зависимостей (pip install -r requirements.txt)")
#         exit(1)