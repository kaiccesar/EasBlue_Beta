import re
import sqlite3
import asyncio
import aiohttp
import hashlib
import logging
import threading
import unicodedata
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from dataclasses import dataclass
from typing import List, Optional
from aiohttp import TCPConnector, ClientTimeout
import requests


# Configuração de adaptadores para datetime
def adapt_datetime(dt):
    return dt.isoformat()


sqlite3.register_adapter(datetime, adapt_datetime)

DEFAULT_BASE_URLS = [
    "https://www.gov.br/pt-br/noticias",
    "https://www.infoescola.com/",
    "https://brasil.un.org/pt-br/sdgs",
    "https://brasilescola.uol.com.br/",
    "https://g1.globo.com/",
    "https://www.folha.uol.com.br/",
    "https://www.bbc.com/portuguese",
    "https://oglobo.globo.com/",
    "https://blog.scielo.org/",
    "https://revistagalileu.globo.com/"
]
ALLOWED_DOMAINS = [
    ".gov.br", ".un.org", "blog.scielo.org",
    "folha.uol.com.br", "g1.globo.com",
    "infoescola.com", "brasilescola.uol.com.br",
    "bbc.com", "oglobo.globo.com", "revistagalileu.globo.com"
]

DEFAULT_CONFIG = {
    "MAX_DEPTH": 2,
    "RATE_LIMIT": 1.5,
    "REQUEST_TIMEOUT": 10,
    "MAX_CONNECTIONS": 30,
    "DB_PATH": "Easblue_Prototipo.db",
    "SAFETY_THRESHOLD": 0.3,

    "ALLOWED_DOMAINS": ALLOWED_DOMAINS
}


def discover_seed_urls(base_url: str, allowed_domains: list = None) -> list:
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Erro ao acessar {base_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    found_urls = set()
    for tag in soup.find_all('a', href=True):
        url = urljoin(base_url, tag['href'])
        if allowed_domains:
            if any(domain.lower() in url.lower() for domain in allowed_domains):
                found_urls.add(url)
        else:
            found_urls.add(url)
    return list(found_urls)


# Modelos de dados
@dataclass
class Page:
    url: str
    title: str
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


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EasBlue")


# Gerenciamento de banco de dados com migrações
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
                last_crawled TEXT,
                domain TEXT NOT NULL,
                checksum TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_tokens ON pages(tokens)",
            "CREATE INDEX IF NOT EXISTS idx_safety ON pages(safety_score)",
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
                    logger.error("Erro na migração: %s", str(e))
            conn.commit()


# Processamento de conteúdo
class ContentProcessor:
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
            'nenhum', 'nenhuma', 'cada', 'outrem', 'tampouco', 'de', 'da', 'do',
            'das', 'dos', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'pelo',
            'pela', 'pelos', 'pelas', 'com', 'sem', 'sob', 'sobre', 'trás', 'ante',
            'até', 'após', 'perante', 'mas', 'e', 'ou', 'então', 'portanto', 'pois',
            'porquanto', 'contudo', 'todavia', 'embora', 'logo', 'porém', 'se',
            'porque', 'assim', 'que', 'qual', 'quem', 'cujo', 'cuja', 'cujos',
            'cujas', 'quando', 'onde', 'como', 'quanto', 'quantos', 'quantas'
        }
        return base_stopwords.union(custom_stopwords)

    def process_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).lower()
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> str:
        words = text.split()
        processed_words = []
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                processed_words.append(self.stemmer.stem(word))
        return ' '.join(processed_words)


# Verificação de segurança e análise de conteúdo
class SecurityEngine:
    @staticmethod
    def generate_checksum(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    async def verify_url(url: str) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(url, timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning("Falha na verificação de URL %s: %s", url, str(e))
            return False

    @classmethod
    def analyze_content(cls, text: str) -> float:
        text = text.lower()
        danger = 0.0
        risk_patterns = [
            (r"\b(vacina .* mata|cloroquina cura)\b", 0.8),
            (r"\b(senha|cpf|cart[ãa]o)\b", 0.3),
            (r"\b(urgente!|promoção)\b", 0.2)
        ]
        for pattern, score in risk_patterns:
            danger += len(re.findall(pattern, text)) * score
        return min(danger, 1.0)


# Sistema de crawling
class CrawlerService:
    def __init__(self):
        self.processor = ContentProcessor()
        logger.info("Crawler inicializado")

    async def crawl_site(self, base_url: str):
        logger.info("Iniciando crawling em: %s", base_url)
        async with aiohttp.ClientSession(
                connector=TCPConnector(limit=DEFAULT_CONFIG["MAX_CONNECTIONS"]),
                timeout=ClientTimeout(total=DEFAULT_CONFIG["REQUEST_TIMEOUT"])
        ) as session:
            queue = [(base_url, 0)]
            visited = set()
            while queue:
                url, depth = queue.pop(0)
                if url in visited or depth > DEFAULT_CONFIG["MAX_DEPTH"]:
                    continue
                visited.add(url)
                if await self._process_page(session, url):
                    new_links = await self._extract_links(session, url, depth)
                    queue.extend(new_links)
                    await asyncio.sleep(DEFAULT_CONFIG["RATE_LIMIT"])

    async def _process_page(self, session, url: str) -> bool:
        try:
            content = await self._fetch_page(session, url)
            if not content:
                return False
            page = self._create_page(url, content)
            if page.safety_score > DEFAULT_CONFIG["SAFETY_THRESHOLD"]:
                logger.debug("Conteúdo inseguro ignorado: %s", url)
                return False
            self._save_page(page)
            logger.info("Página indexada: %s", url)
            return True
        except Exception as e:
            logger.error("Erro no processamento de %s: %s", url, str(e))
            return False

    async def _fetch_page(self, session, url: str) -> Optional[str]:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                logger.warning("Status inválido %s em %s", response.status, url)
                return None
        except Exception as e:
            logger.warning("Falha ao buscar %s: %s", url, str(e))
            return None

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
            last_crawled=datetime.now().isoformat(),
            domain=parsed.netloc,
            checksum=SecurityEngine.generate_checksum(text)
        )

    def _save_page(self, page: Page):
        try:
            conn = DatabaseManager.get_connection()
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pages 
                    (url, title, content, simplified_content, tokens, safety_score, last_crawled, domain, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    page.url,
                    page.title,
                    page.content,
                    page.simplified_content,
                    page.tokens,
                    page.safety_score,
                    page.last_crawled,
                    page.domain,
                    page.checksum
                ))
        except sqlite3.Error as e:
            logger.error("Erro ao salvar página %s: %s", page.url, str(e))

    async def _extract_links(self, session, url: str, depth: int) -> List[tuple]:
        try:
            async with session.get(url) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                return [
                    (urljoin(url, link['href']), depth + 1)
                    for link in soup.find_all('a', href=True)
                    if self._is_valid(urljoin(url, link['href']))
                ]
        except Exception as e:
            logger.warning("Erro ao extrair links de %s: %s", url, str(e))
            return []

    def _is_valid(self, url: str) -> bool:
        parsed = urlparse(url)
        return any(parsed.netloc.endswith(domain) for domain in DEFAULT_CONFIG["ALLOWED_DOMAINS"])


# Motor de busca
class SearchService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.processor = ContentProcessor()
        self._initialize()
        logger.info("Motor de busca pronto. Vocabulário: %d termos",
                    len(self.vectorizer.get_feature_names_out()))

    def _initialize(self):
        try:
            conn = DatabaseManager.get_connection()
            cur = conn.execute("SELECT title FROM pages")
            docs = [row[0] for row in cur.fetchall()]
            if docs:
                logger.info("Treinando modelo com %d títulos", len(docs))
                self.vectorizer.fit(docs)
            else:
                logger.warning("Nenhum título para treinamento")
        except sqlite3.Error as e:
            logger.error("Falha na inicialização: %s", str(e))

    def search(self, query: str) -> List[SearchResult]:
        logger.info("Nova busca: '%s'", query)
        self._initialize()
        processed_query = self._process_query(query)
        if not processed_query:
            logger.warning("Consulta vazia")
            return []
        query_tokens = set(processed_query.split())
        try:
            query_vec = self.vectorizer.transform([processed_query])
        except ValueError:
            logger.error("Consulta contém termos desconhecidos")
            return []
        results = []
        conn = DatabaseManager.get_connection()
        try:
            cur = conn.execute("""
                SELECT url, title, safety_score 
                FROM pages 
                WHERE safety_score < ?
            """, (DEFAULT_CONFIG["SAFETY_THRESHOLD"],))
            rows = cur.fetchall()
            if not rows:
                return []
            docs = [row[1] for row in rows]
            doc_matrix = self.vectorizer.transform(docs)
            similarities = (query_vec * doc_matrix.T).toarray().flatten()
            for i, sim in enumerate(similarities):
                title = rows[i][1]
                processed_title = self.processor.tokenize(self.processor.process_text(title))
                title_tokens = set(processed_title.split())
                if not query_tokens.intersection(title_tokens):
                    continue
                results.append(SearchResult(
                    url=rows[i][0],
                    title=title,
                    snippet=title,
                    relevance=sim,
                    safety_rating="🟢 Seguro" if rows[i][2] < 0.3 else "🔴 Não verificado"
                ))
            logger.info("Busca retornou %d resultados", len(results))
            results.sort(key=lambda x: x.relevance, reverse=True)
            return results[:20]
        except sqlite3.Error as e:
            logger.error("Erro no banco de dados: %s", str(e))
            return []

    def _process_query(self, query: str) -> str:
        return self.processor.tokenize(self.processor.process_text(query))


# Inicialização de dados e conteúdo interno
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
        dynamic_seed_urls = []

        for base in DEFAULT_BASE_URLS:
            discovered = discover_seed_urls(base, ALLOWED_DOMAINS)
            logger.info("Discovered {} URLs from base {}".format(len(discovered), base))
            dynamic_seed_urls.extend(discovered)
        # Remove duplicatas
        dynamic_seed_urls = list(set(dynamic_seed_urls))
        # Executa o crawling para cada URL descoberta em paralelo
        await asyncio.gather(*(crawler.crawl_site(url) for url in dynamic_seed_urls))

    @staticmethod
    def _add_fallback_content():
        content = {
            "easblue://ajuda": (
                "Central de Ajuda EasBlue:\n"
                "1. Digite termos simples na busca\n"
                "2. Verifique o selo de segurança\n"
                "3. Utilize termos específicos\n"
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
                        (url, title, content, simplified_content, tokens, safety_score, last_crawled, domain, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        url,
                        "Ajuda EasBlue",
                        processed,
                        processed[:500],
                        processor.tokenize(processed),
                        0.0,
                        datetime.now().isoformat(),
                        "easblue.internal",
                        SecurityEngine.generate_checksum(processed)
                    ))
            logger.info("Conteúdo interno adicionado")
        except sqlite3.Error as e:
            logger.error("Erro no conteúdo interno: %s", str(e))


# Aplicação web com Flask
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


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


# Tarefas em background para atualização periódica
async def background_tasks():
    logger.info("Iniciando tarefas em background")
    await DataInitializer.initialize()
    while True:
        try:
            crawler = CrawlerService()
            dynamic_seed_urls = []
            for base in DEFAULT_BASE_URLS:
                discovered = discover_seed_urls(base, ALLOWED_DOMAINS)
                dynamic_seed_urls.extend(discovered)
            dynamic_seed_urls = list(set(dynamic_seed_urls))
            await asyncio.gather(*(crawler.crawl_site(url) for url in dynamic_seed_urls))
            logger.info("Ciclo de atualização concluído")
            await asyncio.sleep(3600)  # Atualiza a cada hora
        except Exception as e:
            logger.error("Erro nas tarefas em background: %s", str(e))
            await asyncio.sleep(60)


if __name__ == "__main__":
    DatabaseManager.get_connection()
    threading.Thread(
        target=lambda: asyncio.run(background_tasks()),
        daemon=True
    ).start()
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
