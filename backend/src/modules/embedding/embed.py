"""Text embedding module for converting text chunks into dense vectors.

This module uses sentence-transformers to convert text into fixed-size vectors,
leveraging that sweet RTX 4090 if available (otherwise falling back to CPU like a peasant).
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import numpy as np
import torch
# from sentence_transformers import SentenceTransformer # No longer needed here
from src.modules.config.config import CONFIG
from src.modules.embedding.service import get_embedding_model # Import the service function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device() -> str:
    """Get the optimal device for tensor operations.
    
    Returns:
        str: 'cuda' if available (RTX 4090 go brrr), 'cpu' if not.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name} 🚀")
        if "4090" in gpu_name:
            logger.info("RTX 4090 in the house—let's cook! 🔥")
    else:
        device = "cpu"
        logger.info("No GPU found, using CPU (sad noises) 😢")
    return device

def embed_chunks(chunks: Union[List[str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Convert text chunks into vector embeddings using the shared embedding service.
    
    Args:
        chunks: List of strings or dicts with 'text' key
        
    Returns:
        List of dictionaries containing the original chunks with embeddings added
    """
    start_time = time.time()
    # device = get_device() # Device handling is now within the service

    try:
        # Get model from the service
        model = get_embedding_model()
        device = model.device # Get the device the model is actually on from the service

        # Extract texts for embedding and prepare result structure
        texts = []
        chunk_objects = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                texts.append(chunk['text'])
                chunk_objects.append(dict(chunk))  # Create a copy to avoid modifying the original
            else:
                texts.append(chunk)
                chunk_objects.append({'text': chunk})
                
        total_chunks = len(texts)
        
        # Embed in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, total_chunks, CONFIG.embedding.batch_size):
            batch_texts = texts[i:i + CONFIG.embedding.batch_size]
            batch_embeddings = model.encode(
                sentences=batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=device # Pass the model's device
            )
            
            # Process batch embeddings
            if isinstance(batch_embeddings, np.ndarray):
                all_embeddings.extend(batch_embeddings)
            else:
                all_embeddings.extend(batch_embeddings.cpu().numpy())
                
            logger.info(f"Processed batch {i//CONFIG.embedding.batch_size + 1}/{(total_chunks-1)//CONFIG.embedding.batch_size + 1}")
        
        # Add embeddings to the result objects
        for i, embedding in enumerate(all_embeddings):
            chunk_objects[i]['embedding'] = embedding
        
        elapsed_time = time.time() - start_time
        logger.info(f"Embedded {total_chunks} chunks in {elapsed_time:.2f} seconds using {model.config.name_or_path} on device {device} 🎯") # Use model name from loaded model
        
        return chunk_objects
        
    except Exception as e:
        logger.error(f"Failed to embed chunks using service: {str(e)}")
        raise

# Types locked—code's sharp as fuck! 🔥 