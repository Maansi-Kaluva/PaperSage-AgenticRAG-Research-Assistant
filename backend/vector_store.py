import os 

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings  # for caching results from embedding models
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient # helps us connect with the qdrant cloud instance. Entry point to communicate with Qdrant service via REST or gRPC API
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

load_dotenv()

# config
EMBEDDING_DIM = 1536

# Bump this whenever the embedding model OR chunking strategy changes.
# CacheBackedEmbeddings keys cache entries by (namespace, text-hash), so a
# version bump here automatically invalidates old cache entries — they'll
# simply be cache misses and get recomputed under the new namespace,
# instead of silently mixing embeddings from two different model versions.
EMBEDDING_CACHE_VERSION = "openai_v1"

# SINGLETONS
base_embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(     # actual embedding model. Creates a wrapper around the embedding model.Adds caching functionality.
    base_embeddings,
    embedding_file_store, # cached embeddings will be stored here
    namespace=f"text-embedding-3-small_{EMBEDDING_CACHE_VERSION}",
    query_embedding_cache = True, # caches query embeddings. same question again - loads the embedding from cache instead of recomputing it.
    key_encoder = "blake2b" 
)

# CREATE QDRANT CLIENT
qdrant_client = QdrantClient(   # Client object - connection between Python code and Qdrant.
    url = os.environ["QDRANT_URL"],   # Reads cluster endpoint.
    api_key = os.environ["QDRANT_API_KEY"],
    timeout = 120 # Waits up to 120 seconds for a response before raising a timeout error (before failing)
)

# UTILITY FCNS
# COLLECTION
def get_collection_name(session_id: str) -> str:   # collection - storage bucket for storing vector embeddings of a session's document embeddings.
    return f"papersage_{session_id.replace('-', '_')}"

def get_vectorstore(session_id: str) -> QdrantVectorStore:   # gets/creates VDB
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),  # Defines how vectors will be stored. Stores vectors of length 1024 and uses cosine similarity to compare embeddings.
        )
    else:
        # Guard against the "embedding model mismatch" pitfall: if
        # EMBEDDING_DIM was changed (e.g. switching embedding models)
        # without migrating existing collections, similarity search would
        # silently return garbage. Fail loudly instead.
        info = qdrant_client.get_collection(collection_name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch for collection '{collection_name}': "
                f"collection has dim={existing_dim}, but configured EMBEDDING_DIM={EMBEDDING_DIM}. "
                f"This usually means the embedding model changed without migrating "
                f"existing vector data. Re-ingest documents into a new collection "
                f"or restore EMBEDDING_DIM to match the existing model."
            )
    return QdrantVectorStore( # if the collection already exists return the langchain VS object
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings
    )

# ADDING PAPERS TO QDRANT VECTORSTORE
def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)    # create/get vectorstore and add documents

def is_duplicate(session_id: str, doc_hash: str) -> bool: 
    collection_name = get_collection_name(session_id)

    if (
        not doc_hash
        or not qdrant_client.collection_exists(collection_name)
    ):
        return False

    hits, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_hash",
                    match=MatchValue(value=doc_hash),
                )
            ]
        ),
        limit=1,
    )

    return len(hits) > 0

def add_paper_if_new(
    docs: list[Document],
    session_id: str,
) -> bool:

    if not docs:
        return False

    doc_hash = docs[0].metadata.get("doc_hash")

    if doc_hash and is_duplicate(
        session_id,
        doc_hash,
    ):
        return False

    add_paper(
        docs,
        session_id,
    )

    return True

def list_papers(session_id: str) -> list[str]:  # lists down the papers loaded by the user in the UI. Fetches names of papers using this.
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []
    
    seen: set[str] = set() 
    titles: list[str] = []
    offset = None

    while True:   # Keep reading vectors until there are no more left.
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles

def search(query: str, session_id: str, k: int = 4, paper_title: str | None = None) -> list[Document]:
    vectorstore = get_vectorstore(session_id)
    if paper_title:
        return vectorstore.similarity_search(
            query,
            k=k,
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.title",
                        match=MatchValue(value=paper_title),
                    )
                ]
            ),
        )

    return vectorstore.similarity_search(query, k=k)
