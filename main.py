import asyncio
import hashlib
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import httpx
import unicodedata
from aiohttp import TCPConnector, ClientTimeout
from bs4 import BeautifulSoup
from cachetools import TTLCache
from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pydantic import BaseSettings, ValidationError
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Configuração Avançada ---
class AppSettings(BaseSettings):
    MAX_DEPTH: int = 3
    RATE_LIMIT: float = 1.2
    REQUEST_TIMEOUT: int = 15
    MAX_CONNECTIONS: int = 50
    DB_PATH: str = "easblue_plus.db"
    SAFETY_THRESHOLD: float = 0.25
    CRAWL_INTERVAL: int = 3600
    SEED_URLS: List[str] = [
        "https://www.infoescola.com/",
        "https://www.gov.br/pt-br/noticias",
        "https://brasil.un.org/pt-br/sdgs",
        "https://brasilescola.uol.com.br/"
    ]
    BM25_WEIGHT: float = 0.7
    TFIDF_WEIGHT: float = 0.3
    CACHE_SIZE: int = 1000
    CACHE_TTL: int = 300

    class Config:
        env_file = ".env"
        env_prefix = "EASBLUE_"

try:
    SETTINGS = AppSettings()
except ValidationError as e:
    logging.error(f"Erro de configuração: {e}")
    raise

# --- Modelos de Dados ---
@dataclass
class Page:
    url: str
    content: str
    simplified_content: str
    tokens: str
    safety_score: float
    last_crawled: str
    domain: str
    checksum: str

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    relevance: float
    safety_rating: str
    last_crawled: str

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("easblue.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EasBlue+")

# --- Cache ---
search_cache = TTLCache(maxsize=SETTINGS.CACHE_SIZE, ttl=SETTINGS.CACHE_TTL)

# --- Gerenciamento de Banco de Dados ---
class DatabaseManager:
    _local = threading.local()

    @classmethod
    def get_connection(cls):
        if not hasattr(cls._local, "conn"):
            cls._local.conn = sqlite3.connect(
                SETTINGS.DB_PATH,
                check_same_thread=False,
                timeout=15
            )
            cls._run_migrations(cls._local.conn)
        return cls._local.conn

    @staticmethod
    def _run_migrations(conn):
        migrations = [
            """CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                simplified_content TEXT NOT NULL,
                tokens TEXT NOT NULL,
                safety_score REAL DEFAULT 0,
                last_crawled TEXT,
                domain TEXT NOT NULL,
                checksum TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_tokens ON pages(tokens)",
            "CREATE INDEX IF NOT EXISTS idx_safety ON pages(safety_score)",
            "CREATE INDEX IF NOT EXISTS idx_domain ON pages(domain)",
            """CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )""",
            "INSERT OR IGNORE INTO schema_version (version) VALUES (3)"
        ]
        with conn:
            for migration in migrations:
                try:
                    conn.execute(migration)
                except sqlite3.Error as e:
                    logger.error(f"Erro na migração: {str(e)}")
            conn.commit()

# --- Processamento de Texto Melhorado ---
class AdvancedContentProcessor:
    def __init__(self):
        self._initialize_nltk()
        self.stemmer = SnowballStemmer("portuguese")
        self.stop_words = set(stopwords.words("portuguese"))
        logger.info("Processador de conteúdo avançado inicializado")

    def _initialize_nltk(self):
        try:
            # Tenta carregar as stopwords; se não existirem, baixa-as
            stopwords.words("portuguese")
        except LookupError:
            import nltk
            nltk.download('stopwords')

    def _remove_accents(self, text: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def process_text(self, text: str) -> str:
        text = self._remove_accents(text)
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'[^\w\s]', '', text).lower()

    def tokenize(self, text: str) -> str:
        return ' '.join([
            self.stemmer.stem(word) for word in text.split()
            if word not in self.stop_words and len(word) > 2
        ])

# --- Sistema de Segurança Aprimorado ---
class EnhancedSecurityEngine:
    RISK_PATTERNS = [
        (r"\b(vacina .* mata|cloroquina cura)\b", 0.8),
        (r"\b(senha|cpf|cart[ãa]o)\b", 0.3),
        (r"\b(urgente!|promoção)\b", 0.2)
    ]

    @staticmethod
    def generate_checksum(content: str) -> str:
        return hashlib.sha3_256(content.encode()).hexdigest()

    @staticmethod
    async def check_google_safe_browsing(target_url: str) -> bool:
        api_key = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")
        if not api_key:
            return True

        safe_browsing_url = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
        payload = {
            "client": {"clientId": "EasBlue", "clientVersion": "6.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": target_url}]
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{safe_browsing_url}?key={api_key}",
                    json=payload,
                    timeout=5
                )
                return not bool(response.json().get('matches'))
        except Exception as e:
            logger.error(f"Erro Safe Browsing: {str(e)}")
            return True

    @classmethod
    def analyze_content(cls, text: str) -> float:
        text = text.lower()
        danger = 0.0
        for pattern, score in cls.RISK_PATTERNS:
            danger += len(re.findall(pattern, text)) * score
        return min(danger, 1.0)

# --- Crawler Aprimorado ---
class EnhancedCrawler:
    def __init__(self):
        self.processor = AdvancedContentProcessor()
        self.robots_cache = {}
        logger.info("Crawler avançado inicializado")

    async def _get_robots_txt(self, domain: str) -> Optional[str]:
        if domain in self.robots_cache:
            return self.robots_cache[domain]

        robots_url = f"https://{domain}/robots.txt"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(robots_url, timeout=5)
                if response.status_code == 200:
                    self.robots_cache[domain] = response.text
                    return response.text
        except Exception as e:
            logger.warning(f"Erro ao buscar robots.txt: {str(e)}")
        return None

    async def _can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        robots_txt = await self._get_robots_txt(parsed.netloc)
        if robots_txt:
            from urllib.robotparser import RobotFileParser
            rp = RobotFileParser()
            rp.parse(robots_txt.splitlines())
            return rp.can_fetch("*", url)
        return True

    async def _process_page(self, session: aiohttp.ClientSession, url: str) -> bool:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Falha ao acessar {url}: {response.status}")
                    return False
                content = await response.text()
        except Exception as e:
            logger.error(f"Erro ao buscar página {url}: {str(e)}")
            return False

        # Processamento do conteúdo
        processor = self.processor
        simplified = processor.process_text(content)
        tokens = processor.tokenize(simplified)
        safety_score = EnhancedSecurityEngine.analyze_content(simplified)
        checksum = EnhancedSecurityEngine.generate_checksum(content)
        domain = urlparse(url).netloc
        last_crawled = datetime.now().isoformat()

        # Salva no banco de dados
        conn = DatabaseManager.get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pages (url, content, simplified_content, tokens, safety_score, last_crawled, domain, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (url, content, simplified, tokens, safety_score, last_crawled, domain, checksum))
            logger.info(f"Página processada e salva: {url}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar página {url}: {str(e)}")
            return False

    async def _extract_links(self, session: aiohttp.ClientSession, url: str, depth: int) -> List[tuple]:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                html = await response.text()
        except Exception as e:
            logger.error(f"Erro ao extrair links de {url}: {str(e)}")
            return []

        soup = BeautifulSoup(html, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            link = urljoin(url, tag["href"])
            parsed_link = urlparse(link)
            if parsed_link.scheme in ["http", "https"]:
                links.append((link, depth + 1))
        return links

    async def crawl_site(self, base_url: str):
        logger.info(f"Iniciando crawling avançado em: {base_url}")
        async with aiohttp.ClientSession(
            connector=TCPConnector(limit=SETTINGS.MAX_CONNECTIONS),
            timeout=ClientTimeout(total=SETTINGS.REQUEST_TIMEOUT)
        ) as session:
            queue = [(base_url, 0)]
            visited = set()
            while queue:
                url, depth = queue.pop(0)
                if url in visited or depth > SETTINGS.MAX_DEPTH:
                    continue
                if not await self._can_fetch(url):
                    logger.info(f"Respeitando robots.txt: {url}")
                    continue
                visited.add(url)
                if await self._process_page(session, url):
                    new_links = await self._extract_links(session, url, depth)
                    queue.extend(new_links)
                    await asyncio.sleep(SETTINGS.RATE_LIMIT)

# --- Motor de Busca Híbrido ---
class HybridSearchEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.bm25 = None
        self.corpus = []
        self._initialize()
        logger.info("Motor de busca híbrido pronto")

    def _initialize(self):
        try:
            conn = DatabaseManager.get_connection()
            cur = conn.execute("SELECT tokens FROM pages")
            docs = [row[0] for row in cur.fetchall()]
            if docs:
                logger.info(f"Treinando modelos com {len(docs)} documentos")
                tokenized_docs = [doc.split() for doc in docs]
                self.bm25 = BM25Okapi(tokenized_docs)
                self.tfidf_vectorizer.fit(docs)
                self.corpus = docs
            else:
                logger.warning("Nenhum documento para treinamento")
        except sqlite3.Error as e:
            logger.error(f"Falha na inicialização: {str(e)}")

    def _process_query(self, query: str) -> str:
        processor = AdvancedContentProcessor()
        return processor.process_text(query)

    def _create_result(self, row, score: float) -> SearchResult:
        url, simplified_content, safety_score, last_crawled = row
        title = url  # sem extração específica de título, usa-se a URL como título
        snippet = simplified_content[:200] + "..." if len(simplified_content) > 200 else simplified_content
        safety_rating = "seguro" if safety_score < SETTINGS.SAFETY_THRESHOLD else "não seguro"
        return SearchResult(url=url, title=title, snippet=snippet, relevance=score, safety_rating=safety_rating, last_crawled=last_crawled)

    def search(self, query: str) -> List[SearchResult]:
        cache_key = f"search:{query.lower()}"
        if cache_key in search_cache:
            logger.info(f"Cache hit para: '{query}'")
            return search_cache[cache_key]

        logger.info(f"Nova busca: '{query}'")
        processed = self._process_query(query)
        if not processed:
            return []

        try:
            bm25_scores = self.bm25.get_scores(processed.split()) if self.bm25 else []
            tfidf_scores = self.tfidf_vectorizer.transform([processed])
            results = []
            conn = DatabaseManager.get_connection()
            cur = conn.execute("""
                SELECT url, simplified_content, safety_score, last_crawled 
                FROM pages 
                WHERE safety_score < ?
            """, (SETTINGS.SAFETY_THRESHOLD,))
            rows = cur.fetchall()
            for idx, row in enumerate(rows):
                try:
                    doc_vec = self.tfidf_vectorizer.transform([row[1]])
                    tfidf_score = (tfidf_scores * doc_vec.T).toarray()[0][0]
                    bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0
                    combined_score = (SETTINGS.BM25_WEIGHT * bm25_score +
                                      SETTINGS.TFIDF_WEIGHT * tfidf_score)
                    if combined_score > 0.1:
                        results.append(self._create_result(row, combined_score))
                except Exception as e:
                    logger.error(f"Erro no documento {row[0]}: {str(e)}")
            logger.info(f"Busca retornou {len(results)} resultados")
            sorted_results = sorted(results, key=lambda x: x.relevance, reverse=True)[:20]
            search_cache[cache_key] = sorted_results
            return sorted_results
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}")
            return []

# --- Data Initializer ---
class DataInitializer:
    @staticmethod
    async def initialize():
        conn = DatabaseManager.get_connection()
        cur = conn.execute("SELECT COUNT(*) FROM pages")
        count = cur.fetchone()[0]
        if count == 0:
            logger.info("Inicializando dados com seed URLs")
            crawler = EnhancedCrawler()
            async with aiohttp.ClientSession(
                connector=TCPConnector(limit=SETTINGS.MAX_CONNECTIONS),
                timeout=ClientTimeout(total=SETTINGS.REQUEST_TIMEOUT)
            ) as session:
                for url in SETTINGS.SEED_URLS:
                    await crawler._process_page(session, url)

# --- API REST e Frontend Moderno ---
app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour", "50 per minute"]

)

@app.route("/api/search", methods=["GET"])
@limiter.limit("10 per second")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Consulta vazia"}), 400
    results = HybridSearchEngine().search(query)
    return jsonify([
        {
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "relevance": r.relevance,
            "safety": r.safety_rating,
            "last_crawled": r.last_crawled
        } for r in results
    ])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("q", "").strip()
        results = HybridSearchEngine().search(query)
        return render_template("index.html", results=results, query=query)
    return render_template("index.html")

@app.route("/health")
def health_check():
    try:
        conn = DatabaseManager.get_connection()
        conn.execute("SELECT 1")
        count = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        return jsonify({
            "status": "ok",
            "database": "operacional",
            "indexed_pages": count
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Tarefas em Background ---
async def background_operations():
    logger.info("Iniciando operações em background")
    await DataInitializer.initialize()
    while True:
        try:
            crawler = EnhancedCrawler()
            for url in SETTINGS.SEED_URLS:
                await crawler.crawl_site(url)
            logger.info("Ciclo de atualização concluído")
            await asyncio.sleep(SETTINGS.CRAWL_INTERVAL)
        except Exception as e:
            logger.error(f"Erro nas operações em background: {str(e)}")
            await asyncio.sleep(60)

# --- Ponto de Entrada ---
if __name__ == "__main__":
    try:
        import nltk
        nltk.data.find('corpora/stopwords')
    except LookupError:
        import nltk
        nltk.download('stopwords')

    DatabaseManager.get_connection()
    threading.Thread(
        target=lambda: asyncio.run(background_operations()),
        daemon=True
    ).start()
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
