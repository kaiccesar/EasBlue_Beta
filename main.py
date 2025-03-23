import os
import re
import sqlite3
import asyncio
import aiohttp
import hashlib
import logging
import threading
import unicodedata
import numpy as np
import dns.resolver
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import deque
from itertools import chain
from bs4 import BeautifulSoup
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pybloom_live import BloomFilter

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Configurações padrões atualizadas para selecionar sites mais confiáveis e factuais
DEFAULT_CONFIG = {
    "MAX_DEPTH": int(os.getenv("MAX_DEPTH", "2")),
    "RATE_LIMIT": float(os.getenv("RATE_LIMIT", "1.5")),
    "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", "10")),
    "MAX_CONNECTIONS": int(os.getenv("MAX_CONNECTIONS", "30")),
    "DB_PATH": os.getenv("DB_PATH", "Easblue_Prototipo.db"),
    "SAFETY_THRESHOLD": float(os.getenv("SAFETY_THRESHOLD", "0.3")),
    "BLOOM_FILTER_CAPACITY": 500000,
    "BLOOM_ERROR_RATE": 0.001
}

USE_JS_RENDERING = os.getenv("USE_JS_RENDERING", "False").lower() == "true"
USE_ADVANCED_NLP = os.getenv("USE_ADVANCED_NLP", "False").lower() == "true"

sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())

# Sites base com foco em fontes institucionais, governamentais e factuais
DEFAULT_BASE_URLS = [
    "https://www.gov.br/pt-br/noticias",
    "https://www.ibge.gov.br",
    "https://portal.fiocruz.br",
    "https://www.in.gov.br",
    "https://www.ipea.gov.br",
    "https://www.bcb.gov.br",
    "https://www.un.org/en",
    "https://www.who.int",
    "https://www.cdc.gov",
    "https://www.scielo.br",
    "https://www.nature.com",
    "https://www.science.org"
]

# Domínios permitidos, focando em fontes seguras e institucionais
ALLOWED_DOMAINS = [
    "gov.br", "ibge.gov.br", "fiocruz.br", "in.gov.br", "ipea.gov.br", "bcb.gov.br",
    "un.org", "who.int", "cdc.gov", "scielo.br", "nature.com", "science.org"
]


@dataclass
class Page:
    url: str
    title: str
    content: str
    simplified_content: str
    tokens: str
    safety_score: float
    trust_score: float
    fact_check_rating: str
    last_crawled: str
    domain: str
    checksum: str
    reported: int = 0


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    relevance: float
    safety_rating: str
    trust_rating: str
    fact_check_rating: str


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EasBlue")


class DatabaseManager:
    _local = threading.local()

    @classmethod
    def get_connection(cls):
        if not hasattr(cls._local, "conn"):
            cls._local.conn = sqlite3.connect(
                DEFAULT_CONFIG["DB_PATH"],
                check_same_thread=False
            )
            cls._run_migrations(cls._local.conn)
        return cls._local.conn

    @staticmethod
    def _run_migrations(conn):
        migrations = [
            """CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                simplified_content TEXT NOT NULL,
                tokens TEXT NOT NULL,
                safety_score REAL DEFAULT 0,
                trust_score REAL DEFAULT 1.0,
                fact_check_rating TEXT DEFAULT 'Não verificado',
                last_crawled TEXT,
                domain TEXT NOT NULL,
                checksum TEXT NOT NULL,
                reported INTEGER DEFAULT 0
            )""",
            "CREATE INDEX IF NOT EXISTS idx_tokens ON pages(tokens)",
            "CREATE INDEX IF NOT EXISTS idx_safety ON pages(safety_score)",
            "CREATE INDEX IF NOT EXISTS idx_trust ON pages(trust_score)",
            "CREATE INDEX IF NOT EXISTS idx_reported ON pages(reported)",
            """CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )""",
            "INSERT OR IGNORE INTO schema_version (version) VALUES (4)",
            """CREATE TABLE IF NOT EXISTS autocomplete_terms (
                term TEXT PRIMARY KEY,
                count INTEGER DEFAULT 1
            )""",
            "CREATE INDEX IF NOT EXISTS idx_autocomplete_term ON autocomplete_terms(term)"
        ]

        additional_migrations = [
            "ALTER TABLE pages ADD COLUMN trust_score REAL DEFAULT 1.0",
            "ALTER TABLE pages ADD COLUMN fact_check_rating TEXT DEFAULT 'Não verificado'",
            "ALTER TABLE pages ADD COLUMN reported INTEGER DEFAULT 0"
        ]

        with conn:
            for migration in migrations:
                try:
                    conn.execute(migration)
                except sqlite3.Error as err:
                    logger.error(f"Erro na migração: {err}")

            for migration in additional_migrations:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError as err:
                    if "duplicate column name" not in str(err):
                        logger.error(f"Erro na migração adicional: {err}")
                except sqlite3.Error as err:
                    logger.error(f"Erro na migração adicional: {err}")
            conn.commit()


class ContentProcessor:
    REGEX_PATTERNS = {
        'hyphen': re.compile(r'-'),
        'urls': re.compile(r'https?://\S+'),
        'numbers': re.compile(r'\b\d+\b'),
        'specials': re.compile(r'[^\w\s]', flags=re.UNICODE),
        'spaces': re.compile(r'\s+')
    }

    def __init__(self):
        self.stemmer = SnowballStemmer("portuguese")
        self.stop_words = self._load_stopwords()
        logger.info("Processador de conteúdo inicializado")

    def _load_stopwords(self):
        base_stopwords = set(stopwords.words("portuguese"))
        custom_stopwords = {
            'também', 'após', 'segundo', 'mesmo', 'outro', 'outra', 'outros',
            'outras', 'etc', 'aquele', 'aquela', 'isso', 'esse', 'essa', 'disso',
            'desse', 'dessa', 'entre', 'sobre', 'através', 'quando', 'onde', 'como',
            'porque', 'porquê', 'pois', 'assim', 'agora', 'ainda', 'já', 'antes',
            'depois', 'sempre', 'nunca', 'muito', 'pouco', 'muitos', 'poucos',
            'mais', 'menos', 'qual', 'quais', 'qualquer', 'algum', 'alguma',
            'alguns', 'algumas', 'todo', 'toda', 'todos', 'todas', 'nada', 'ninguém',
            'nenhum', 'nenhuma', 'cada', 'outrem', 'tampouco'
        }
        return base_stopwords.union(custom_stopwords)

    def process_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).lower()
        text = self.REGEX_PATTERNS['hyphen'].sub(' ', text)
        text = self.REGEX_PATTERNS['urls'].sub('', text)
        text = self.REGEX_PATTERNS['numbers'].sub('', text)
        text = self.REGEX_PATTERNS['specials'].sub('', text)
        text = self.REGEX_PATTERNS['spaces'].sub(' ', text).strip()
        return text

    def tokenize(self, text: str) -> str:
        return ' '.join(
            self.stemmer.stem(word) for word in text.split() if len(word) > 2 and word not in self.stop_words)


class SecurityEngine:
    # Ampliação dos padrões de risco com novos termos associados a desinformação e promessas milagrosas
    RISK_PATTERNS = [
        (r"\b(vacina .* mata|cloroquina cura)\b", 0.8),
        (r"\b(senha|cpf|cart[ãa]o)\b", 0.3),
        (r"\b(urgente!|promoção)\b", 0.2),
        (r"\b(conspira[cç][aã]o|fake news|desinformação)\b", 0.5),
        (r"\b(milagre|cura milagrosa|cura instantânea)\b", 0.7),
        (r"\b(alerta vermelho|emergência)\b", 0.4)
    ]

    @classmethod
    def analyze_content(cls, text: str) -> float:
        danger = 0.0
        text = text.lower()
        for pattern, score in cls.RISK_PATTERNS:
            danger += len(re.findall(pattern, text)) * score
        return min(danger, 1.0)

    @staticmethod
    def generate_checksum(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()


class CrawlerService:
    def __init__(self):
        self.processor = ContentProcessor()
        self.visited_filter = BloomFilter(
            capacity=DEFAULT_CONFIG["BLOOM_FILTER_CAPACITY"],
            error_rate=DEFAULT_CONFIG["BLOOM_ERROR_RATE"]
        )
        self.lock = threading.Lock()
        logger.info("Crawler inicializado com filtro Bloom")

    async def crawl_site(self, base_url: str):
        logger.info(f"Iniciando crawling em: {base_url}")
        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=DEFAULT_CONFIG["MAX_CONNECTIONS"]),
                timeout=aiohttp.ClientTimeout(total=DEFAULT_CONFIG["REQUEST_TIMEOUT"])
        ) as session:
            queue: deque[Tuple[str, int]] = deque([(base_url, 0)])
            while queue:
                url, depth = queue.popleft()
                with self.lock:
                    if url in self.visited_filter or depth > DEFAULT_CONFIG["MAX_DEPTH"]:
                        continue
                    self.visited_filter.add(url)
                if not await self._should_process(url):
                    continue
                success = await self._process_page(session, url)
                if success:
                    new_links = await self._extract_links(session, url, depth)
                    queue.extend(new_links)
                    await asyncio.sleep(DEFAULT_CONFIG["RATE_LIMIT"])

    async def _should_process(self, url: str) -> bool:
        if not is_valid_dns(url):
            logger.debug(f"DNS inválido: {url}")
            return False
        parsed = urlparse(url)
        if not any(parsed.netloc.endswith(domain) for domain in ALLOWED_DOMAINS):
            logger.debug(f"Domínio não permitido: {parsed.netloc}")
            return False
        return True

    async def _process_page(self, session, url: str) -> bool:
        try:
            content = await self._fetch_page(session, url)
            if not content:
                return False
            page = self._create_page(url, content)
            if page.safety_score > DEFAULT_CONFIG["SAFETY_THRESHOLD"] or page.trust_score < 0.6:
                logger.debug(f"Conteúdo ignorado: {url}")
                return False
            self._save_page(page)
            logger.info(f"Página indexada: {url}")
            return True
        except Exception as err:
            logger.error(f"Erro no processamento de {url}: {err}")
            return False

    async def _fetch_page(self, session, url: str) -> Optional[str]:
        if USE_JS_RENDERING:
            return await self._render_js_page(url)
        retries = 3
        delay = 1
        for attempt in range(retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    logger.warning(f"Status {response.status} para {url}")
            except Exception as err:
                logger.warning(f"Tentativa {attempt + 1} falhou para {url}: {err}")
            await asyncio.sleep(delay)
            delay *= 2
        return None

    async def _render_js_page(self, url: str) -> Optional[str]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._render_with_selenium, url)
        except Exception as err:
            logger.error(f"Erro no rendering JS para {url}: {err}")
            return None

    def _render_with_selenium(self, url: str) -> str:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        # Removido o ChromeDriverManager: utiliza apenas a configuração padrão
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        content = driver.page_source
        driver.quit()
        return content

    def _create_page(self, url: str, content: str) -> Page:
        soup = BeautifulSoup(content, 'html.parser')
        text = self.processor.process_text(soup.get_text())
        parsed = urlparse(url)
        title_tag = soup.find("title")
        title = title_tag.string.strip() if title_tag and title_tag.string else parsed.netloc
        return Page(
            url=url,
            title=title,
            content=text,
            simplified_content=text[:500],
            tokens=self.processor.tokenize(text),
            safety_score=SecurityEngine.analyze_content(text),
            trust_score=assess_trust(text),
            fact_check_rating=fact_check_url(url),
            last_crawled=datetime.now().isoformat(),
            domain=parsed.netloc,
            checksum=SecurityEngine.generate_checksum(text),
            reported=0
        )

    def _save_page(self, page: Page):
        try:
            conn = DatabaseManager.get_connection()
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO pages 
                    (url, title, content, simplified_content, tokens, 
                     safety_score, trust_score, fact_check_rating, 
                     last_crawled, domain, checksum, reported)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        page.url,
                        page.title,
                        page.content,
                        page.simplified_content,
                        page.tokens,
                        page.safety_score,
                        page.trust_score,
                        page.fact_check_rating,
                        page.last_crawled,
                        page.domain,
                        page.checksum,
                        page.reported
                    )
                )
                title_terms = self._extract_autocomplete_terms(page.title)
                for term_item in title_terms:
                    conn.execute(
                        """INSERT INTO autocomplete_terms (term, count)
                        VALUES (?, 1)
                        ON CONFLICT(term) DO UPDATE SET count = count + 1""",
                        (term_item,)
                    )
        except sqlite3.Error as err:
            logger.error(f"Erro ao salvar página {page.url}: {err}")

    def _extract_autocomplete_terms(self, text: str) -> List[str]:
        normalized = unicodedata.normalize('NFKD', text).lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        words = re.findall(r'\b\w{3,}\b', normalized)
        return [word for word in words if word not in self.processor.stop_words]

    async def _extract_links(self, session, url: str, depth: int) -> List[Tuple[str, int]]:
        try:
            content = await self._fetch_page(session, url)
            if not content:
                return []
            soup = BeautifulSoup(content, 'html.parser')
            links: List[Tuple[str, int]] = []
            for link in soup.find_all('a', href=True):
                new_url = urljoin(url, link['href'])
                if self._is_valid(new_url) and is_valid_dns(new_url):
                    links.append((new_url, depth + 1))
            return links
        except Exception as err:
            logger.warning(f"Erro ao extrair links de {url}: {err}")
            return []

    @staticmethod
    def _is_valid(url: str) -> bool:
        scheme = urlparse(url).scheme
        return scheme in ['http', 'https']


def is_valid_dns(url: str) -> bool:
    domain = urlparse(url).netloc
    try:
        dns.resolver.resolve(domain, 'A')
        return True
    except Exception:
        return False


def assess_trust(text: str) -> float:
    suspicious_words = ["exclusivo", "bombástico", "urgente", "alerta", "chocante"]
    score = 1.0
    for word in suspicious_words:
        if word in text.lower():
            score -= 0.15
    return max(score, 0.0)


def fact_check_url(url: str) -> str:
    trusted_domains = [
        "gov.br", "ibge.gov.br", "fiocruz.br", "in.gov.br", "ipea.gov.br", "bcb.gov.br",
        "un.org", "who.int", "cdc.gov", "scielo.br", "nature.com", "science.org"
    ]
    domain = urlparse(url).netloc.lower()
    return "Verificado" if any(domain.endswith(td) for td in trusted_domains) else "Não verificado"


class SearchService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.85)
        self.processor = ContentProcessor()
        self._initialize()
        logger.info("Motor de busca otimizado pronto")

    def _initialize(self):
        try:
            conn = DatabaseManager.get_connection()
            cur = conn.execute("SELECT title FROM pages")
            docs = [row[0] for row in cur.fetchall()]
            if docs:
                logger.info(f"Treinando modelo com {len(docs)} títulos")
                self.vectorizer.fit(docs)
        except sqlite3.Error as err:
            logger.error(f"Falha na inicialização: {err}")

    def search(self, query: str) -> List[SearchResult]:
        logger.info(f"Nova busca: '{query}'")
        self._initialize()
        processed_query = self._process_query(query)
        if not processed_query:
            return []
        try:
            query_vec = self.vectorizer.transform([processed_query])
        except ValueError:
            return []
        results = []
        conn = DatabaseManager.get_connection()
        try:
            cur = conn.execute(
                """SELECT url, title, safety_score, trust_score, fact_check_rating 
                FROM pages 
                WHERE safety_score < ? AND reported = 0""",
                (DEFAULT_CONFIG["SAFETY_THRESHOLD"],)
            )
            rows = cur.fetchall()
            if not rows:
                return []
            docs = [row[1] for row in rows]
            doc_matrix = self.vectorizer.transform(docs)
            similarities = np.dot(doc_matrix, query_vec.T).toarray().flatten()
            query_tokens = set(processed_query.split())
            for i, sim in enumerate(similarities):
                title = rows[i][1]
                processed_title = self.processor.tokenize(self.processor.process_text(title))
                title_tokens = set(processed_title.split())
                if not query_tokens.intersection(title_tokens):
                    continue
                safety_rating = "🟢 Seguro" if rows[i][2] < 0.3 else "🔴 Não verificado"
                trust_value = rows[i][3]
                if trust_value >= 0.8:
                    trust_rating = "✅ Confiável"
                elif trust_value < 0.6:
                    trust_rating = "⚠️ Suspeito"
                else:
                    trust_rating = "ℹ️ Mediano"
                results.append(SearchResult(
                    url=rows[i][0],
                    title=title,
                    snippet=title[:150] + "..." if len(title) > 150 else title,
                    relevance=sim,
                    safety_rating=safety_rating,
                    trust_rating=trust_rating,
                    fact_check_rating=rows[i][4]
                ))
            trust_order = {"✅ Confiável": 2, "ℹ️ Mediano": 1, "⚠️ Suspeito": 0}
            results.sort(key=lambda x: (x.relevance, trust_order.get(x.trust_rating, 1)), reverse=True)
            return results[:15]
        except sqlite3.Error as err:
            logger.error(f"Erro no banco de dados: {err}")
            return []

    def _process_query(self, query: str) -> str:
        return self.processor.tokenize(self.processor.process_text(query))


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.getenv("SECRET_KEY", "super_secret_key_123")


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = SearchService().search(query)
            logger.info(f"Consulta '{query}' retornou {len(results)} resultados")
    return render_template("index.html",
                           results=results,
                           query=query,
                           results_count=len(results))


@app.route("/report", methods=["POST"])
def report():
    reported_url = request.form.get("url")
    if reported_url:
        try:
            conn = DatabaseManager.get_connection()
            with conn:
                conn.execute("UPDATE pages SET reported = 1 WHERE url = ?", (reported_url,))
            flash("Conteúdo reportado com sucesso!", "success")
        except Exception:
            flash("Erro ao reportar o conteúdo", "danger")
    return redirect(url_for("index"))


@app.route('/autocomplete')
def autocomplete():
    def normalize_text(text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii').lower()

    raw_term = request.args.get('term', '')
    normalized_term = normalize_text(raw_term)
    if not normalized_term:
        return jsonify([])
    try:
        conn = DatabaseManager.get_connection()
        cur = conn.execute(
            """SELECT term FROM autocomplete_terms
            WHERE term LIKE ? || '%'
            ORDER BY count DESC
            LIMIT 10""",
            (normalized_term,)
        )
        suggestions = [row[0] for row in cur.fetchall()]
        return jsonify(suggestions)
    except sqlite3.Error as err:
        logger.error(f"Erro na busca de autocompletar: {err}")
        return jsonify([])


class DataInitializer:
    @staticmethod
    async def initialize():
        conn = DatabaseManager.get_connection()
        if conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0:
            logger.info("Inicializando banco de dados...")
            await DataInitializer.seed_content()
            DataInitializer.add_fallback_content()

    @staticmethod
    async def seed_content():
        crawler = CrawlerService()
        dynamic_seed_urls = await DataInitializer.discover_seed_urls()
        logger.info(f"Iniciando indexação de {len(dynamic_seed_urls)} URLs")
        batch_size = 10
        for i in range(0, len(dynamic_seed_urls), batch_size):
            batch = dynamic_seed_urls[i:i + batch_size]
            await asyncio.gather(*(crawler.crawl_site(url) for url in batch))

    @staticmethod
    async def discover_seed_urls() -> List[str]:
        async with aiohttp.ClientSession() as session:
            tasks = [discover_seed_urls_async(session, base) for base in DEFAULT_BASE_URLS]
            results = await asyncio.gather(*tasks)
            return list(set(chain.from_iterable(results)))

    @staticmethod
    def add_fallback_content():
        content = {
            "easblue://ajuda": (
                "Central de Ajuda EasBlue:\n"
                "1. Utilize termos simples na busca\n"
                "2. Verifique sempre a fonte dos resultados\n"
                "3. Denuncie conteúdos suspeitos\n"
                "Exemplo: 'direitos do idoso'"
            )
        }
        try:
            conn = DatabaseManager.get_connection()
            processor = ContentProcessor()
            with conn:
                for url, text in content.items():
                    processed = processor.process_text(text)
                    conn.execute(
                        """INSERT OR IGNORE INTO pages 
                        (url, title, content, simplified_content, tokens, 
                         safety_score, trust_score, fact_check_rating, 
                         last_crawled, domain, checksum, reported)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            url,
                            "Ajuda EasBlue",
                            processed,
                            processed[:500],
                            processor.tokenize(processed),
                            0.0,
                            1.0,
                            "Verificado",
                            datetime.now().isoformat(),
                            "easblue.internal",
                            SecurityEngine.generate_checksum(processed),
                            0
                        )
                    )
            logger.info("Conteúdo interno adicionado")
        except sqlite3.Error as err:
            logger.error(f"Erro no conteúdo interno: {err}")


async def background_tasks():
    logger.info("Iniciando tarefas em background")
    await DataInitializer.initialize()
    semaphore = asyncio.Semaphore(DEFAULT_CONFIG["MAX_CONNECTIONS"])

    async def limited_crawl(url, crawl_service):
        async with semaphore:
            await crawl_service.crawl_site(url)

    while True:
        try:
            crawl_service = CrawlerService()
            dynamic_seed_urls = await DataInitializer.discover_seed_urls()
            tasks = [limited_crawl(url, crawl_service) for url in dynamic_seed_urls]
            await asyncio.gather(*tasks)
            logger.info("Ciclo de atualização concluído")
            await asyncio.sleep(3600)
        except Exception as err:
            logger.error(f"Erro nas tarefas em background: {err}")
            await asyncio.sleep(60)


async def discover_seed_urls_async(session, base_url: str) -> List[str]:
    try:
        async with session.get(base_url, timeout=10) as response:
            content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')
            found_urls = set()
            for tag in soup.find_all('a', href=True):
                url = urljoin(base_url, tag['href'])
                if any(domain.lower() in url.lower() for domain in ALLOWED_DOMAINS):
                    found_urls.add(url)
            return list(found_urls)
    except Exception as err:
        logger.error(f"Erro ao descobrir URLs: {err}")
        return []


if __name__ == "__main__":
    DatabaseManager.get_connection()
    threading.Thread(
        target=lambda: asyncio.run(background_tasks()),
        daemon=True
    ).start()
    app.run(host='0.0.0.0', port=5000, debug=False, ssl_context='adhoc')
