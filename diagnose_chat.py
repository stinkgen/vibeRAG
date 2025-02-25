"""Diagnostic script to test chat functionality directly."""

import asyncio
import json
from config.config import CONFIG
from generation.generate import chat_with_knowledge
from vector_store.milvus_ops import connect_milvus
from embedding.embed import embed_chunks
from setup_test_data import setup_test_data
from retrieval.search import semantic_search

async def test_chat_directly():
    """Test chat functionality directly."""
    # Print config info
    print(f"Chat config:")
    print(f"  Model: {CONFIG.chat.model}")
    print(f"  Provider: {CONFIG.chat.provider}")
    print(f"  Temperature: {CONFIG.chat.temperature}")
    
    if hasattr(CONFIG.chat, 'ollama_url'):
        print(f"  Ollama URL: {CONFIG.chat.ollama_url}")
    else:
        print("  No Ollama URL configured!")
        
    # Set up test data
    print("\nSetting up test data...")
    setup_test_data()
    
    # Test semantic search
    print("\nTesting semantic search directly...")
    search_results = semantic_search("What is VibeRAG?", filename="test.txt")
    print(f"Search returned {len(search_results)} results")
    for i, result in enumerate(search_results):
        print(f"Result {i+1}:")
        for key, value in result.items():
            if key == 'embedding':
                print(f"  {key}: [numpy array of shape {value.shape}]")
            else:
                print(f"  {key}: {value}")
    
    # Test chat with test.txt
    print("\nTesting chat with 'What is VibeRAG?'...")
    try:
        response_gen = chat_with_knowledge(
            query="What is VibeRAG?",
            filename="test.txt",
            knowledge_only=True
        )
        
        print("Response chunks:")
        response = ""
        count = 0
        async for chunk in response_gen:
            count += 1
            print(f"  Chunk {count}: {chunk}")
            response += chunk
            
        print(f"\nTotal chunks: {count}")
        print(f"Full response: {response}")
    except Exception as e:
        print(f"Error during chat: {str(e)}")
    
    # Try with OpenAI explicitly
    print("\nTrying with OpenAI explicitly...")
    try:
        response_gen = chat_with_knowledge(
            query="What is VibeRAG?",
            filename="test.txt",
            knowledge_only=True,
            provider="openai",
            model="gpt-3.5-turbo"
        )
        
        print("Response chunks:")
        response = ""
        count = 0
        async for chunk in response_gen:
            count += 1
            print(f"  Chunk {count}: {chunk}")
            response += chunk
            
        print(f"\nTotal chunks: {count}")
        print(f"Full response: {response}")
    except Exception as e:
        print(f"Error with OpenAI: {str(e)}")
        
    # Try with Ollama explicitly
    print("\nTrying with Ollama explicitly...")
    try:
        response_gen = chat_with_knowledge(
            query="What is VibeRAG?",
            filename="test.txt",
            knowledge_only=True,
            provider="ollama",
            model=CONFIG.ollama.model  # Use the config value instead of hardcoding
        )
        
        print("Response chunks:")
        response = ""
        count = 0
        async for chunk in response_gen:
            count += 1
            print(f"  Chunk {count}: {chunk}")
            response += chunk
            
        print(f"\nTotal chunks: {count}")
        print(f"Full response: {response}")
    except Exception as e:
        print(f"Error with Ollama: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_chat_directly()) 