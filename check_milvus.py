import asyncio
from vector_store.milvus_ops import connect_milvus, init_collection
from config.config import CONFIG

async def main():
    # Connect to Milvus (synchronous)
    connect_milvus()
    collection = init_collection()
    
    # Use collection.query to search for test data
    expr = f"{CONFIG.milvus.filename_field} == 'test.txt'"
    print(f"Running query with expression: {expr}")
    
    # Query directly without requiring vector search
    results = collection.query(
        expr=expr,
        output_fields=["text", "metadata", "filename", "tags"],
        limit=10
    )
    
    print(f'Query results: {results}')
    
    if not results:
        print("No results found for filename='test.txt'")
    else:
        print(f"Found {len(results)} documents with filename='test.txt'")

if __name__ == "__main__":
    asyncio.run(main()) 