import os
import re
from rag.vector_store import VectorStore

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

def extract_metadata(filename: str, content: str) -> dict:
    """Extract metadata from filename and content."""
    metadata = {
        "source_file": filename,
        "doc_type": "unknown",
        "date": "unknown",
        "node_id": "unknown",
        "customer_id": "unknown",
        "severity": "unknown"
    }

    # Extract doc_type from prefix
    if filename.startswith("PM"):
        metadata["doc_type"] = "post_mortem"
    elif filename.startswith("MV"):
        metadata["doc_type"] = "maintenance_visit"
    elif filename.startswith("PROC"):
        metadata["doc_type"] = "procedure"
    elif filename.startswith("VB"):
        metadata["doc_type"] = "vendor_bulletin"
    elif filename.startswith("CP"):
        metadata["doc_type"] = "capacity_planning"

    # Extract date from filename if present (e.g., PM-2026-001)
    date_match = re.search(r'2026-\d{2}-\d{2}|2026-Q[1-4]', content)
    if not date_match:
        date_match = re.search(r'2026', filename)
    if date_match:
        metadata["date"] = date_match.group(0)

    # Extract Node ID from content
    node_match = re.search(r'Node\s+(\d+)', content, re.IGNORECASE)
    if node_match:
        metadata["node_id"] = f"Node {node_match.group(1)}"

    # Extract Severity from content if present
    severity_match = re.search(r'P[1-4]\s+\((.*?)\)', content)
    if severity_match:
        metadata["severity"] = severity_match.group(0)

    return metadata

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks of approx chunk_size words."""
    words = text.split()
    chunks = []
    if not words:
        return chunks
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def ingest_corpus():
    """Reads corpus, chunks, extracts metadata, and upserts to ChromaDB."""
    print("Starting ingestion pipeline...")
    store = VectorStore()
    
    chunks_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for filename in os.listdir(CORPUS_DIR):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(CORPUS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        metadata = extract_metadata(filename, content)
        doc_chunks = chunk_text(content)
        
        for i, chunk in enumerate(doc_chunks):
            chunk_id = f"{filename}_chunk_{i}"
            chunks_to_add.append(chunk)
            metadatas_to_add.append(metadata)
            ids_to_add.append(chunk_id)

    print(f"Adding {len(chunks_to_add)} chunks to ChromaDB...")
    store.add_chunks(chunks_to_add, metadatas_to_add, ids_to_add)
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_corpus()
