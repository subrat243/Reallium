"""
Database migration script.
"""

from sqlalchemy import create_engine
from database.database import Base
from database.models import User, Detection, Feedback, ModelVersion, AuditLog, ThreatIntelligence
from api.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migrations():
    """Run database migrations."""
    logger.info("Starting database migrations...")
    
    # Create engine
    engine = create_engine(settings.DATABASE_URL)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    logger.info("Database migrations completed successfully")
    logger.info(f"Created tables: {', '.join(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    run_migrations()
