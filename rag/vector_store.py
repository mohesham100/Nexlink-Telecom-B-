import os
import chromadb
from chromadb.utils import embedding_functions

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "nexlink_noc_docs"

class VectorStore:
    """
    Wrapper for ChromaDB vector store used in Nexlink Telecom NOC RAG.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' for embeddings.
    """
    def __init__(self):
        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Setup SentenceTransformerEmbeddingFunction
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.collection = self.get_or_create_collection()

    def get_or_create_collection(self):
        """
        Gets or creates the ChromaDB collection with HNSW index config.
        """
        # ANN index config and metadata index are handled by ChromaDB defaults (HNSW)
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity for HNSW index
        )

    def add_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        """
        Adds text chunks and their metadata payload to the vector store.
        """
        if not chunks:
            return
            
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 5, where_filter: dict = None) -> dict:
        """
        Queries the vector store using text.
        Supports metadata filtering with ChromaDB's $and/$or/$eq operators.
        """
        query_args = {
            "query_texts": [query_text],
            "n_results": n_results
        }
        if where_filter:
            query_args["where"] = where_filter

        results = self.collection.query(**query_args)
        return results

    def query_with_embedding(self, embedding: list[float], n_results: int = 5, where_filter: dict = None) -> dict:
        """
        Queries the vector store using a pre-computed embedding.
        """
        query_args = {
            "query_embeddings": [embedding],
            "n_results": n_results
        }
        if where_filter:
            query_args["where"] = where_filter

        results = self.collection.query(**query_args)
        return results
