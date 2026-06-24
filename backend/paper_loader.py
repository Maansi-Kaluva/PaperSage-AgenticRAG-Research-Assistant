import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def _stamp_title(docs: list[Document], title: str) -> list[Document]:   # Included PageIndex
    for idx, doc in enumerate(docs):
        doc.metadata["title"] = title

        doc.metadata.setdefault(
            "page_index",
            idx,
        )

        doc.metadata.setdefault(
            "page_number",
            doc.metadata.get("page", idx),
        )

    return docs

def _compute_doc_hash(docs: list[Document]) -> str:
    full_text = "".join(
        d.page_content
        for d in docs
    )

    return hashlib.sha256(
        full_text.encode("utf-8")
    ).hexdigest()


def _stamp_doc_hash(
    docs: list[Document],
) -> list[Document]:

    if not docs:
        return docs

    doc_hash = _compute_doc_hash(docs)

    for doc in docs:
        doc.metadata["doc_hash"] = doc_hash

    return docs

def load_pdf(file_path: str) -> list[Document]:   # Loads a PDF from disk using the file_path
    docs = PyMuPDFLoader(file_path).load()
    return _stamp_doc_hash(
        _stamp_title(
            _splitter.split_documents(docs),
            Path(file_path).stem,
        )
    )


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_doc_hash(
        _stamp_title(
            _splitter.split_documents(docs),
            Path(file_path).stem,
        )
    )


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_doc_hash(
        _stamp_title(
            _md_splitter.split_documents(docs),
            Path(file_path).stem,
        )
    )


def load_docx(file_path: str) -> list[Document]:
    docs = Docx2txtLoader(file_path).load()
    return _stamp_doc_hash(
        _stamp_title(
            _splitter.split_documents(docs),
            Path(file_path).stem,
        )
    )


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _stamp_doc_hash(
        _stamp_title(
            _splitter.split_documents(docs),
            title,
        )
    )


# downloading arXiv papers is done using their ids
def _extract_arxiv_id(query: str) -> str | None:
    query = query.strip()
    m = _ARXIV_ID_RE.search(query)
    if not m:
        return None
    return re.sub(
        r"v\d+$",
        "",
        m.group(1),
    )


def _arxiv_api_lookup(arxiv_id: str) -> str:    # Given an arXiv ID (in the sidebar), fetch paper title.
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)   # extracts all titles.
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:   # User gives the title -> finds matching arxiv id
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:  # arxiv id -> download pdf ->chunk it -> return docs 
    """Download and chunk an ArXiv paper PDF by its bare ID."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()   # raw binary data
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:   # delete - false coz it automatically deletes after the file is closed
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        docs = PyMuPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        return _stamp_doc_hash(
            _stamp_title(
                _splitter.split_documents(docs),
                title,
            )
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)   # deleting the path after the processing is finished


def load_arxiv(query: str) -> list[Document]:   # wrapper function for loading using arxiv id
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:  # wrapper fcn which handles all our types of documents
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    if ext == ".docx":
        return load_docx(source)
    raise ValueError(f"Unsupported file type: {ext}")