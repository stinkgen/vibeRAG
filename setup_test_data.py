"""Script to manually set up test data in Milvus for testing."""

import json
import numpy as np
from vector_store.milvus_ops import connect_milvus, init_collection
from embedding.embed import embed_chunks
from config.config import CONFIG

# Test chunks with text about VibeRAG for testing 
TEST_CHUNKS = [
    {
        'text': 'VibeRAG is a powerful AI research assistant that combines retrieval-augmented generation with a sleek user interface.',
        'metadata': {
            'filename': 'test.txt',
            'page': 1,
            'category': 'documentation'
        }
    },
    {
        'text': 'VibeRAG features include document chat, presentation generation, and comprehensive research reports based on your data.',
        'metadata': {
            'filename': 'test.txt',
            'page': 2,
            'category': 'documentation'
        }
    }
]

def setup_test_data():
    """Set up test data in Milvus for testing."""
    print("\n*** Setting up test data in Milvus ***")
    
    # Connect to Milvus
    connect_milvus()
    print("Connected to Milvus")
    
    # Initialize collection
    collection = init_collection()
    print("Initialized collection")
    
    # Generate embeddings for test chunks
    texts = [chunk['text'] for chunk in TEST_CHUNKS]
    print(f"Embedding {len(texts)} text chunks")
    embeddings = embed_chunks(texts)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Combine embeddings with chunks
    chunks_with_embeddings = []
    for i, (chunk, embedding) in enumerate(zip(TEST_CHUNKS, embeddings)):
        chunks_with_embeddings.append({
            'text': chunk['text'],
            'metadata': chunk['metadata'],
            'embedding': embedding,
            'tags': ['test']
        })
    
    # Prepare data for Milvus
    doc_ids = []
    text_list = []
    embedding_list = []
    metadata_list = []
    tags_list = []
    filename_list = []
    category_list = []
    
    for i, chunk in enumerate(chunks_with_embeddings):
        doc_ids.append(f"test_doc_{i}")
        text_list.append(chunk['text'])
        embedding_list.append(chunk['embedding'])
        metadata_list.append(json.dumps(chunk['metadata']))
        tags_list.append(['test'])
        filename_list.append('test.txt')
        category_list.append(chunk['metadata'].get('category', ''))
    
    # Insert data
    insert_data = [
        doc_ids,
        embedding_list,
        text_list,
        metadata_list,
        tags_list,
        filename_list,
        category_list
    ]
    
    print(f"Inserting {len(doc_ids)} items into Milvus")
    collection.insert(insert_data)
    collection.flush()
    print("Data inserted and flushed")
    
    # Verify data was stored
    results = collection.query(
        expr=f"{CONFIG.milvus.filename_field} == 'test.txt'",
        output_fields=["text", "metadata", "filename"],
        limit=10
    )
    
    if not results:
        print("WARNING: No data found after insertion!")
    else:
        print(f"Success! Found {len(results)} items with filename='test.txt'")
        print(f"First result: {results[0]}")

if __name__ == "__main__":
    setup_test_data() 