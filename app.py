from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import logging
from database import VectorDAO, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Economics RAG Chat")
vector_store = VectorDAO()
openai_client = OpenAI()

SYSTEM_PROMPT = """You are an economics expert assistant. Use the provided context to answer the user's question.
Your answers should be:
1. Accurate and based on the provided context
2. Clear and concise
3. Include relevant economic concepts and terminology when appropriate

If the provided context doesn't contain enough information to answer the question fully, acknowledge this and answer based on what is available.
Always maintain a professional and educational tone.

Context:
{context}

User Question: {question}"""

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using OpenAI API."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_response(question: str, context_chunks: List[tuple[str, str, int]]) -> str:
    """Generate response using OpenAI chat completion."""
    # Format context for the prompt
    formatted_context = "\n\n".join([
        f"From {filename} (chunk {chunk_idx}):\n{content}"
        for content, filename, chunk_idx in context_chunks
    ])

    # Create the complete prompt
    prompt = SYSTEM_PROMPT.format(context=formatted_context, question=question)

    # Get completion from OpenAI
    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    vector_store.close()
    logger.info("Vector store connection closed")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that implements RAG-based responses.
    """
    logger.info(f"Received question: {request.message}")

    try:
        # Get embedding for the question
        query_vector = get_embedding(request.message)

        # Search for similar chunks
        similar_chunks = await vector_store.search_similar(query_vector, limit=3)

        if not similar_chunks:
            return ChatResponse(
                response="I apologize, but I couldn't find any relevant information to answer your question.",
                sources=[]
            )

        # Extract content and metadata for response generation
        context_chunks = [
            (chunk.content, chunk.filename, chunk.chunk_index)
            for chunk, score in similar_chunks
        ]

        # Generate response using LLM
        response_text = generate_response(request.message, context_chunks)

        # Format sources
        sources = [f"{chunk.filename} (chunk {chunk.chunk_index})" for chunk, _ in similar_chunks]

        return ChatResponse(
            response=response_text,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)