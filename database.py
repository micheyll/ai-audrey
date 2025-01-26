from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from typing import List, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create SQLAlchemy base class
Base = declarative_base()

# Database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./economics.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class DocumentChunk(Base):
    """Document chunk model for storing text chunks and their vector mappings."""
    __tablename__ = "document_chunks"

    id: int = Column(Integer, primary_key=True, index=True)
    content: str = Column(Text, nullable=False)
    filename: str = Column(String(255), nullable=False, index=True)
    chunk_index: int = Column(Integer, nullable=False)
    # Using String type for SQLite compatibility, but enforcing UUID format in application code
    vector_id: str = Column(String(36), nullable=False, unique=True)

    def __init__(self, **kwargs):
        if 'vector_id' in kwargs:
            # Validate UUID format
            try:
                uuid.UUID(kwargs['vector_id'])
            except ValueError:
                raise ValueError("vector_id must be a valid UUID string")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, filename='{self.filename}', chunk_index={self.chunk_index})>"

class VectorDAO:
    """Data Access Object for vector operations using Qdrant and SQLite."""

    COLLECTION_NAME = "document_embeddings"
    VECTOR_SIZE = 1536  # OpenAI embedding dimension
    QDRANT_PATH = "./qdrant_data"  # Local persistent storage

    def __init__(self):
        """Initialize VectorDAO with Qdrant client."""
        self.client = QdrantClient(path=self.QDRANT_PATH)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Ensure the vector collection exists, create if it doesn't."""
        collections = self.client.get_collections().collections
        if not any(collection.name == self.COLLECTION_NAME for collection in collections):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
            )

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self.client.close()

    @contextmanager
    def _get_db(self) -> Session:
        """Get database session."""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def check_file_exists(self, filename: str) -> bool:
        """
        Check if a file has already been ingested.

        Args:
            filename: Name of the file to check

        Returns:
            bool: True if file exists in database
        """
        with self._get_db() as db:
            exists = db.query(DocumentChunk).filter_by(filename=filename).first() is not None
            return exists

    async def delete_file_chunks(self, filename: str) -> None:
        """
        Delete all chunks associated with a file from both SQLite and Qdrant.

        Args:
            filename: Name of the file to delete
        """
        with self._get_db() as db:
            # Get all chunks for the file
            chunks = db.query(DocumentChunk).filter_by(filename=filename).all()

            if not chunks:
                return

            # Delete vectors from Qdrant - use string UUIDs directly
            vector_ids = [chunk.vector_id for chunk in chunks]  # Already string UUIDs
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=models.PointIdsList(
                    points=vector_ids
                )
            )

            # Delete chunks from SQLite
            db.query(DocumentChunk).filter_by(filename=filename).delete()
            db.commit()

            logger.info(f"Deleted {len(chunks)} chunks for file: {filename}")

    async def store_document_chunks(self, chunks: List[DocumentChunk], vectors: List[List[float]], replace_existing: bool = True) -> None:
        """
        Store document chunks and their vectors.

        Args:
            chunks: List of DocumentChunk objects
            vectors: List of embedding vectors for each chunk
            replace_existing: If True, replace existing chunks for the same file
        """
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors")

        if not chunks:
            return

        filename = chunks[0].filename

        # Check for existing file
        if replace_existing and self.check_file_exists(filename):
            logger.info(f"File {filename} already exists, deleting old chunks...")
            await self.delete_file_chunks(filename)

        with self._get_db() as db:
            # Store chunks in SQLite
            for chunk in chunks:
                db.add(chunk)
            db.flush()

            # Prepare points for Qdrant
            points = []
            for chunk, vector in zip(chunks, vectors):
                points.append(models.PointStruct(
                    id=chunk.vector_id,  # Already a string UUID
                    vector=vector,
                    payload={
                        "filename": chunk.filename,
                        "chunk_index": chunk.chunk_index
                    }
                ))

            # Store vectors in Qdrant
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )

            # Commit SQLite changes
            db.commit()
            logger.info(f"Stored {len(chunks)} chunks for file: {filename}")

    async def search_similar(self, query_vector: List[float], limit: int = 5) -> List[tuple[DocumentChunk, float]]:
        """
        Search for similar vectors and return their associated chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return

        Returns:
            List of tuples containing (DocumentChunk, similarity_score)
        """
        # Search vectors in Qdrant
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

        # Get associated chunks from SQLite
        response = []
        with self._get_db() as db:
            for hit in results:
                chunk = db.query(DocumentChunk).filter_by(vector_id=hit.id).first()
                if chunk:  # Only add if chunk is found
                    response.append((chunk, hit.score))

        return response

def init_db() -> None:
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)