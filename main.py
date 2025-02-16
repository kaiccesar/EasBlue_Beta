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
from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from dataclasses import dataclass
from typing import List, Optional
from pybloom_live import BloomFilter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import dns.resolver

# Cria um resolvedor personalizado
resolver = dns.resolver.Resolver(configure=False)
resolver.nameservers = ['8.8.8.8']  # Define servidores DNS

try:
    resposta = resolver.resolve('google.com', 'A')
    for registro in resposta:
        print(registro)
except dns.resolver.NXDOMAIN:
    print("Domínio não encontrado")
except dns.resolver.NoAnswer:
    print("Sem resposta para a consulta")

# Configurações dinâmicas
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




# Adaptador para datetime
def adapt_datetime(dt):
    return dt.isoformat()


sqlite3.register_adapter(datetime, adapt_datetime)

# URLs e domínios permitidos
DEFAULT_BASE_URLS = [
    "https://www.gov.br/pt-br/noticias",
    "https://www.gov.br/anatel/pt-br",  # Agência Nacional de Telecomunicações :cite[5]
    "https://www.ibge.gov.br",  # Instituto Brasileiro de Geografia e Estatística
    "https://portal.fiocruz.br",  # Fundação Oswaldo Cruz
    "https://www12.senado.leg.br/noticias",  # Portal oficial do Senado
    "https://www.camara.leg.br",  # Câmara dos Deputados
    "https://agenciabrasil.ebc.com.br",  # Agência Brasil (Empresa Brasil de Comunicação)
    "https://www.scielo.br",  # Biblioteca científica eletrônica
    "https://www.cetic.br",  # Centro Regional de Estudos para o Desenvolvimento da Sociedade da Informação
    "https://www.bcb.gov.br",  # Banco Central do Brasil
    "https://www.ipea.gov.br",  # Instituto de Pesquisa Econômica Aplicada
    "https://www.mctic.gov.br",  # Ministério da Ciência, Tecnologia e Inovações
    "https://www.in.gov.br"  # Diário Oficial da União
]

ALLOWED_DOMAINS = [
    ".gov.br", ".ebc.com.br", "scielo.br", "cetic.br",
    "ibge.gov.br", "fiocruz.br", "bcb.gov.br", "ipea.gov.br",
    "mctic.gov.br", "senado.leg.br", "camara.leg.br", "in.gov.br",
    # Mantendo os anteriores
    ".un.org", "blog.scielo.org", "folha.uol.com.br",
    "g1.globo.com", "infoescola.com", "brasilescola.uol.com.br",
    "bbc.com", "oglobo.globo.com", "revistagalileu.globo.com"
]

# Modelos de dados
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


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EasBlue")


# Gerenciamento de banco de dados
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
            "INSERT OR IGNORE INTO schema_version (version) VALUES (4)"
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
                except sqlite3.Error as e:
                    logger.error("Erro na migração: %s", str(e))

            for migration in additional_migrations:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        logger.error("Erro na migração adicional: %s", str(e))
                except sqlite3.Error as e:
                    logger.error("Erro na migração adicional: %s", str(e))

            conn.commit()


# Processamento de conteúdo
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
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                processed_words.append(self.stemmer.stem(word))
        return ' '.join(processed_words)


# Verificação de segurança
class SecurityEngine:
    RISK_PATTERNS = [
        (r"\b(vacina .* mata|cloroquina cura)\b", 0.8),
        (r"\b(senha|cpf|cart[ãa]o)\b", 0.3),
        (r"\b(urgente!|promoção)\b", 0.2)
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


# Sistema de crawling
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
        logger.info("Iniciando crawling em: %s", base_url)
        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=DEFAULT_CONFIG["MAX_CONNECTIONS"]),
                timeout=aiohttp.ClientTimeout(total=DEFAULT_CONFIG["REQUEST_TIMEOUT"])
        ) as session:
            queue = deque([(base_url, 0)])

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
            logger.debug("DNS inválido: %s", url)
            return False

        parsed = urlparse(url)
        if not any(parsed.netloc.endswith(domain) for domain in ALLOWED_DOMAINS):
            logger.debug("Domínio não permitido: %s", parsed.netloc)
            return False

        return True

    async def _process_page(self, session, url: str) -> bool:
        try:
            content = await self._fetch_page(session, url)
            if not content:
                return False
            page = self._create_page(url, content)

            if page.safety_score > DEFAULT_CONFIG["SAFETY_THRESHOLD"] or page.trust_score < 0.6:
                logger.debug("Conteúdo ignorado: %s", url)
                return False

            self._save_page(page)
            logger.info("Página indexada: %s", url)
            return True
        except Exception as e:
            logger.error("Erro no processamento de %s: %s", url, str(e))
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
                    logger.warning("Status %s para %s", response.status, url)
            except Exception as e:
                logger.warning("Tentativa %d falhou para %s: %s", attempt + 1, url, e)
            await asyncio.sleep(delay)
            delay *= 2
        return None

    async def _render_js_page(self, url: str) -> Optional[str]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._render_with_selenium, url)
        except Exception as e:
            logger.error("Erro no rendering JS para %s: %s", url, e)
            return None

    def _render_with_selenium(self, url: str) -> str:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
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
                conn.execute("""
                    INSERT OR REPLACE INTO pages 
                    (url, title, content, simplified_content, tokens, 
                     safety_score, trust_score, fact_check_rating, 
                     last_crawled, domain, checksum, reported)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
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
                ))
        except sqlite3.Error as e:
            logger.error("Erro ao salvar página %s: %s", page.url, str(e))

    async def _extract_links(self, session, url: str, depth: int) -> List[tuple]:
        try:
            content = await self._fetch_page(session, url)
            if not content:
                return []

            soup = BeautifulSoup(content, 'html.parser')
            links = []
            for link in soup.find_all('a', href=True):
                new_url = urljoin(url, link['href'])
                if self._is_valid(new_url) and is_valid_dns(new_url):
                    links.append((new_url, depth + 1))
            return links
        except Exception as e:
            logger.warning("Erro ao extrair links de %s: %s", url, str(e))
            return []

    def _is_valid(self, url: str) -> bool:
        scheme = urlparse(url).scheme
        return scheme in ['http', 'https']


# Funções auxiliares
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
        "gov.br", "ebc.com.br", "scielo.br", "cetic.br",
        "ibge.gov.br", "fiocruz.br", "bcb.gov.br", "ipea.gov.br",
        "mctic.gov.br", "senado.leg.br", "camara.leg.br", "in.gov.br",
        "un.org", "scielo.org", "bbc.com"
    ]
    domain = urlparse(url).netloc.lower()
    return "Verificado" if any(domain.endswith(td) for td in trusted_domains) else "Não verificado"


# Motor de busca
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
                logger.info("Treinando modelo com %d títulos", len(docs))
                self.vectorizer.fit(docs)
        except sqlite3.Error as e:
            logger.error("Falha na inicialização: %s", str(e))

    def search(self, query: str) -> List[SearchResult]:
        logger.info("Nova busca: '%s'", query)
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
            cur = conn.execute("""
                SELECT url, title, safety_score, trust_score, fact_check_rating 
                FROM pages 
                WHERE safety_score < ? AND reported = 0
            """, (DEFAULT_CONFIG["SAFETY_THRESHOLD"],))

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
                trust_rating = ("✅ Confiável" if trust_value >= 0.8
                                else "⚠️ Suspeito" if trust_value < 0.6
                else "ℹ️ Mediano")

                results.append(SearchResult(
                    url=rows[i][0],
                    title=title,
                    snippet=title[:150] + "..." if len(title) > 150 else title,
                    relevance=sim,
                    safety_rating=safety_rating,
                    trust_rating=trust_rating,
                    fact_check_rating=rows[i][4]
                ))

            results.sort(key=lambda x: (x.relevance, x.trust_rating), reverse=True)
            return results[:15]

        except sqlite3.Error as e:
            logger.error("Erro no banco de dados: %s", str(e))
            return []

    def _process_query(self, query: str) -> str:
        return self.processor.tokenize(self.processor.process_text(query))


# Aplicação Flask
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
            logger.info("Consulta '%s' retornou %d resultados", query, len(results))
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
        except Exception as e:
            flash("Erro ao reportar o conteúdo", "danger")
    return redirect(url_for("index"))


# Inicialização de dados
class DataInitializer:
    @staticmethod
    async def initialize():
        conn = DatabaseManager.get_connection()
        if conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0:
            logger.info("Inicializando banco de dados...")
            await DataInitializer._seed_content()
            DataInitializer._add_fallback_content()

    @staticmethod
    async def _seed_content():
        crawler = CrawlerService()
        dynamic_seed_urls = await DataInitializer._discover_urls_async()
        logger.info("Iniciando indexação de %d URLs", len(dynamic_seed_urls))

        batch_size = 10
        for i in range(0, len(dynamic_seed_urls), batch_size):
            batch = dynamic_seed_urls[i:i + batch_size]
            await asyncio.gather(*(crawler.crawl_site(url) for url in batch))

    @staticmethod
    async def _discover_urls_async():
        async with aiohttp.ClientSession() as session:
            tasks = [discover_seed_urls_async(session, base) for base in DEFAULT_BASE_URLS]
            results = await asyncio.gather(*tasks)
            return list(set(chain.from_iterable(results)))

    @staticmethod
    def _add_fallback_content():
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
                    conn.execute("""
                        INSERT OR IGNORE INTO pages 
                        (url, title, content, simplified_content, tokens, 
                         safety_score, trust_score, fact_check_rating, 
                         last_crawled, domain, checksum, reported)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
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
                    ))
            logger.info("Conteúdo interno adicionado")
        except sqlite3.Error as e:
            logger.error("Erro no conteúdo interno: %s", str(e))


# Tarefas em background
async def background_tasks():
    logger.info("Iniciando tarefas em background")
    await DataInitializer.initialize()

    semaphore = asyncio.Semaphore(DEFAULT_CONFIG["MAX_CONNECTIONS"])

    async def limited_crawl(url, crawler):
        async with semaphore:
            await crawler.crawl_site(url)

    while True:
        try:
            crawler = CrawlerService()
            dynamic_seed_urls = await DataInitializer._discover_urls_async()

            tasks = [limited_crawl(url, crawler) for url in dynamic_seed_urls]
            await asyncio.gather(*tasks)

            logger.info("Ciclo de atualização concluído")
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error("Erro nas tarefas em background: %s", str(e))
            await asyncio.sleep(60)


async def discover_seed_urls_async(session, base_url: str):
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
    except Exception as e:
        logger.error("Erro ao descobrir URLs: %s", str(e))
        return []


if __name__ == "__main__":
    DatabaseManager.get_connection()
    threading.Thread(
        target=lambda: asyncio.run(background_tasks()),
        daemon=True
    ).start()
    app.run(host="0.0.0.0", port=5000, use_reloader=False)