from ingestion.ingest import parse_document
from embedding.embed import embed_chunks
from vector_store.milvus_ops import connect_milvus, init_collection, store_embeddings
from retrieval.search import semantic_search
from generation.generate import chat_with_knowledge, create_presentation
from research.crew import research_task
from dotenv import load_dotenv
import logging
import os

# Load env vars—point to root explicitly
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env.local"))
logging.basicConfig(level=logging.INFO)

def run_pipeline(file_path: str) -> None:
    """Run the full ingest-embed-store-retrieve-generate-research pipeline—let's fuckin' go."""
    # Connect to Milvus
    logging.info("Hitting up Milvus—stand by!")
    connect_milvus()

    # Parse the doc
    logging.info(f"Parsing {file_path}—let's get those chunks!")
    chunks = parse_document(file_path)
    logging.info(f"Got {len(chunks)} chunks—ready to embed!")

    # Embed the chunks
    embedded = embed_chunks(chunks)
    logging.info(f"Embedded {len(embedded)} chunks—vectors dropping hot!")

    # Set up Milvus and store
    init_collection(recreate=True)
    store_embeddings(embedded)
    logging.info("Chunks are chilling in Milvus—storage vibes locked in!")

    # Test retrieval
    logging.info("Testing semantic search—let's see what we got!")
    results = semantic_search("platform features")
    if results:
        logging.info(f"Found {len(results)} chunks: {results[0]['text'][:50]}...")
    else:
        logging.info("No hits on platform features—might need more juice!")

    # Test generation with knowledge only
    logging.info("Chatting with the knowledge—give me something good!")
    chat_response = chat_with_knowledge("What's in whitepaper.pdf?", filename="whitepaper.pdf", knowledge_only=True)
    logging.info(f"Chat says: {chat_response[:100]}...")

    # Test generation with web search
    logging.info("Chatting with web search—let's hit Google!")
    chat_response_web = chat_with_knowledge("What's the platform's architecture?", filename="whitepaper.pdf", use_web=True)
    logging.info(f"Chat with web says: {chat_response_web}")  # Full response

    # Test presentation
    logging.info("Cooking up a presentation—slide time!")
    slides = create_presentation("Make a deck about the platform's features")
    logging.info(f"Got {len(slides['slides'])} slides: {slides['slides'][0]['title']} - {slides['slides'][0]['content'][:50]}...")

    # Test research
    logging.info("Research crew's on the case—deep dive incoming!")
    report = research_task("What are the platform's key capabilities?")
    logging.info(f"Research report's in: {report['report'][:100]}...")

if __name__ == "__main__":
    run_pipeline("whitepaper.pdf")