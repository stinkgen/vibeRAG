"""Embedding Service Singleton.

Ensures the SentenceTransformer model is loaded only once.
"""

import logging
from sentence_transformers import SentenceTransformer
from src.modules.config.config import CONFIG
import torch

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            logger.info('Creating the EmbeddingService singleton instance.')
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _get_device(self) -> str:
        """Determine the optimal device for embedding."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} - Using CUDA for embeddings.")
            if "4090" in gpu_name:
                 logger.info("RTX 4090 detected - Prepare for liftoff! 🔥")
        else:
            device = "cpu"
            logger.info("No GPU detected - Using CPU for embeddings.")
        return device

    def _load_model(self):
        """Loads the SentenceTransformer model."""
        if self._model is None:
            model_name = CONFIG.embedding.model_name
            device = self._get_device()
            logger.info(f"Loading SentenceTransformer model: {model_name} onto device: {device}")
            try:
                self._model = SentenceTransformer(model_name, device=device)
                logger.info(f"Successfully loaded embedding model: {model_name}")
            except Exception as e:
                logger.exception(f"Failed to load embedding model {model_name}: {e}")
                # Depending on desired behavior, could raise here or allow fallback
                self._model = None 

    def get_model(self) -> SentenceTransformer:
        """Returns the loaded SentenceTransformer model instance."""
        if self._model is None:
            # Attempt to reload if loading failed initially? Or just raise?
            logger.error("Embedding model is not available.")
            raise RuntimeError("Embedding model could not be loaded.")
        return self._model

# Instantiate the singleton
embedding_service = EmbeddingService()

def get_embedding_model() -> SentenceTransformer:
    """Convenience function to get the embedding model."""
    return embedding_service.get_model() 