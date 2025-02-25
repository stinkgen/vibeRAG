import asyncio
from generation.generate import chat_with_knowledge

async def test_openai():
    """Test OpenAI chat functionality."""
    print("Testing OpenAI chat with 'What is VibeRAG?'...")
    
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

if __name__ == "__main__":
    asyncio.run(test_openai()) 