import asyncio
from pypdf import PdfReader
import tiktoken
from openai import OpenAI, __version__ as openai_version
import os
from typing import List, Tuple
import logging
import uuid
import argparse
from dotenv import load_dotenv
from database import VectorDAO, DocumentChunk, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize OpenAI client with explicit API key
openai_client = OpenAI(api_key=api_key)

def load_pdf(file_path: str) -> str:
    """Load and extract text from PDF."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
        
    logger.info(f"Loading PDF from {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for i, page in enumerate(reader.pages):
        text += page.extract_text() + "\n"
    logger.info(f"Successfully extracted text from {len(reader.pages)} pages")
    return text

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: The target size of each chunk in characters (default: 1000)
        overlap: The number of characters to overlap between chunks (default: 500)
    """
    logger.info(f"Splitting text into chunks of {chunk_size} characters with {overlap} character overlap")
    logger.info(f"Total characters in text: {len(text)}")

    chunks = []
    start_idx = 0

    while start_idx < len(text):
        # Calculate end index for current chunk
        end_idx = min(start_idx + chunk_size, len(text))
        
        # Get the chunk text
        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)
        
        # Move start_idx forward by chunk_size - overlap
        start_idx += chunk_size - overlap
        
        # Prevent infinite loop if overlap is too large
        if start_idx <= 0:
            start_idx = end_idx

    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def get_embeddings(chunks: List[str]) -> List[List[float]]:
    """Get embeddings for text chunks using OpenAI API."""
    logger.info("Generating embeddings using OpenAI API")
    logger.info(f"Using OpenAI Python SDK version: {openai_version}")

    embeddings = []

    try:
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            try:
                response = openai_client.embeddings.create(
                    model="text-embedding-3-large",
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Ingest a PDF file into the vector database.')
    parser.add_argument('filename', type=str, help='Path to the PDF file to ingest')
    parser.add_argument('--chunk-size', type=int, default=1000, 
                      help='Size of text chunks in tokens (default: 1000)')
    parser.add_argument('--overlap', type=int, default=200,
                      help='Number of tokens to overlap between chunks (default: 200)')
    args = parser.parse_args()

    logger.info("Starting document ingestion process")

    # Initialize database
    logger.info("Initializing database")
    init_db()

    # Initialize VectorDAO
    vector_store = VectorDAO()
    logger.info("Initialized vector store")

    try:
        # Load and process PDF
        text = load_pdf(args.filename)
        chunks = split_into_chunks(text, chunk_size=args.chunk_size, overlap=args.overlap)

        # Get embeddings
        embeddings = get_embeddings(chunks)

        # Create DocumentChunk objects
        logger.info("Creating document chunk objects")
        doc_chunks = []
        base_filename = os.path.basename(args.filename)
        for i, chunk in enumerate(chunks):
            # Generate a UUID for each chunk
            vector_id = str(uuid.uuid4())
            doc_chunks.append(DocumentChunk(
                content=chunk,
                filename=base_filename,
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