# src/retriever.py
# Hybrid retriever that combines BM25 and Cosmos DB vector search

from azure.cosmos import CosmosClient
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from config.settings import COSMOS_URI, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER
from src.cosmos_db import query_vector_search
from src.embeddings import generate_embedding
from src.cosmos_db import get_cosmos_container


import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)



class HybridRetriever:
    def __init__(self, top_k_bm25: int = 20, top_k_vec: int = 15):
        # Initialize Cosmos DB client and fetch all stored chunks
        client = CosmosClient(url=COSMOS_URI, credential=COSMOS_KEY)
        db = client.get_database_client(COSMOS_DATABASE)
        container = db.get_container_client(COSMOS_CONTAINER)
        items = list(container.read_all_items())

        # Prepare BM25 index
        self.items = items
        # self.corpus_tokens = [word_tokenize(itm['content'].lower()) for itm in items]
        self.corpus_tokens = [
            word_tokenize(
                (itm.get('content') or itm.get('recommendation') or '').lower()
            )
            for itm in items
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.top_k_bm25 = top_k_bm25
        self.top_k_vec = top_k_vec

    def retrieve(self, query: str):
        # BM25 retrieval
        tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        best_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k_bm25]
        bm25_hits = [self.items[i] for i in best_idxs]

        # Vector retrieval
        q_vec = generate_embedding(query)
        vec_hits = query_vector_search(q_vec, top_k=self.top_k_vec)

        # Merge, preserve order, dedupe
        seen = set()
        hybrid = []
        for hit in bm25_hits + vec_hits:
            uid = hit.get('id')
            if uid not in seen:
                seen.add(uid)
                hybrid.append(hit)
        return hybrid


# Singleton instance for app-wide reuse
_retriever = HybridRetriever()

def hybrid_retrieve(query: str):
    """
    Returns a merged list of top BM25 and vector search hits for the given query.
    Each hit is a dict with keys: id, document_name, section, content, (and score for vector hits).
    """
    return _retriever.retrieve(query)

def get_recommendations_by_section(section=None, subsection=None, doc_name=None):
    container = get_cosmos_container()
    query = "SELECT * FROM c WHERE 1=1"
    params = []
    if section:
        query += " AND LOWER(c.section) = @section"
        params.append({"name": "@section", "value": section.lower()})
    if subsection:
        query += " AND LOWER(c.subsection) = @subsection"
        params.append({"name": "@subsection", "value": subsection.lower()})
    if doc_name:
        query += " AND c.document_name = @doc_name"
        params.append({"name": "@doc_name", "value": doc_name})
    return list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
