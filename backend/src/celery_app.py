from celery import Celery
import logging
from src.modules.config.config import CONFIG # Import our config

logger = logging.getLogger(__name__)

celery_app = Celery(
    'viberag_tasks', # Name of the celery app (can be anything)
    broker=CONFIG.celery.broker_url,
    backend=CONFIG.celery.result_backend,
    include=['src.modules.agent_service.tasks'] # List of modules containing tasks
)

# Load Celery configuration from our config object
celery_app.conf.update(
    task_serializer=CONFIG.celery.task_serializer,
    result_serializer=CONFIG.celery.result_serializer,
    accept_content=CONFIG.celery.accept_content,
    timezone=CONFIG.celery.timezone,
    enable_utc=CONFIG.celery.enable_utc,
    # Add any other Celery config variables you want to set
    # Example: task_acks_late=True
)

logger.info(f"Celery app initialized with broker: {CONFIG.celery.broker_url}")

# Optional: Define base task class for common behavior (e.g., DB session handling)
# class BaseTaskWithDB(celery_app.Task):
#     def __call__(self, *args, **kwargs):
#         db = SessionLocal() # Assuming SessionLocal is accessible here
#         try:
#             return super().__call__(*args, db=db, **kwargs)
#         finally:
#             db.close() 