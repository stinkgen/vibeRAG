from ingestion.ingest import parse_document
from embedding.embed import embed_chunks
import logging

logging.basicConfig(level=logging.INFO)
chunks = parse_document("test.txt")
embedded = embed_chunks(chunks)
logging.info(f"Embedded {len(embedded)} chunks, first vector shape: {len(embedded[0]['embedding'])}")
