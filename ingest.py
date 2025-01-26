import asyncio
from pypdf import PdfReader
import tiktoken
from openai import OpenAI, __version__ as openai_version
import os
from typing import List, Tuple
import logging
import uuid
from database import VectorDAO, DocumentChunk, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> str:
    """Load and extract text from PDF."""
    logger.info(f"Loading PDF from {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += page.extract_text() + "\n"
    logger.info(f"Successfully extracted text from {len(reader.pages)} pages")
    return text

def split_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately chunk_size tokens."""
    logger.info(f"Splitting text into chunks of {chunk_size} tokens")
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    logger.info(f"Total tokens in text: {len(tokens)}")

    chunks = []
    current_chunk: List[int] = []
    current_size = 0

    for token in tokens:
        current_chunk.append(token)
        current_size += 1

        if current_size >= chunk_size:
            chunk_text = enc.decode(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

    # Add the last chunk if it's not empty
    if current_chunk:
        chunk_text = enc.decode(current_chunk)
        chunks.append(chunk_text)

    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def get_embeddings(chunks: List[str]) -> List[List[float]]:
    """Get embeddings for text chunks using OpenAI API."""
    logger.info("Generating embeddings using OpenAI API")
    logger.info(f"Using OpenAI Python SDK version: {openai_version}")

    # Initialize client (will automatically use OPENAI_API_KEY environment variable)
    client = OpenAI()
    embeddings = []

    try:
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                raise
    except Exception as e:
        logger.error("Failed to generate embeddings")
        raise

    logger.info(f"Successfully generated {len(embeddings)} embeddings")
    return embeddings

async def main():
    logger.info("Starting document ingestion process")

    # Initialize database
    logger.info("Initializing database")
    init_db()

    # Initialize VectorDAO
    vector_store = VectorDAO()
    logger.info("Initialized vector store")

    try:
        # Load and process PDF
        pdf_path = "principles_of_economics.pdf"
        text = load_pdf(pdf_path)
        chunks = split_into_chunks(text)

        # Get embeddings
        embeddings = get_embeddings(chunks)

        # Create DocumentChunk objects
        logger.info("Creating document chunk objects")
        doc_chunks = []
        for i, chunk in enumerate(chunks):
            # Generate a UUID for each chunk
            vector_id = str(uuid.uuid4())
            doc_chunks.append(DocumentChunk(
                content=chunk,
                filename="principles_of_economics.pdf",
                chunk_index=i,
                vector_id=vector_id
            ))

        # Store chunks and vectors
        logger.info("Storing chunks and vectors in database")
        await vector_store.store_document_chunks(doc_chunks, embeddings)
        logger.info(f"Successfully processed and stored {len(chunks)} chunks from the PDF")

    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Closing vector store connection")
        vector_store.close()

if __name__ == "__main__":
    asyncio.run(main())