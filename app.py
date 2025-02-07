from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import logging
from database import VectorDAO, init_db
import os
from dotenv import load_dotenv

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

app = FastAPI(title="AI Audrey")
vector_store = VectorDAO()

# Initialize OpenAI clients for different purposes
openai_client = OpenAI(api_key=api_key)  # For embeddings with actual OpenAI
ollama_client = OpenAI(
    base_url="http://llm.anime.world:11434/v1",
    api_key="ollama"  # required but unused by Ollama
)

SYSTEM_PROMPT = """You are a specialized AI assistant focused on analyzing and presenting biographical information of Audrey Hepburn. Your primary function is to work with retrieved biographical content to provide accurate, well-structured responses about Audrey Hepburn, her achievements, and historical context.

When responding:
1. Always maintain biographical accuracy and cite specific life dates/periods
2. Present information chronologically when describing life events
3. Distinguish between verified facts and disputed/uncertain claims
4. Maintain an objective, neutral tone while discussing controversial topics
5. Consider historical context and cultural perspectives of the time period
6. Highlight key achievements, contributions, and historical significance
7. Cross-reference related historical figures when relevant

Format your responses to:
1. Begin with a clear introduction of the person
2. Structure information into logical life periods/phases
3. Use proper names, titles, and dates consistently
4. Include relevant quotes when available
5. Cite sources when presenting specific claims
6. Conclude with a summary of historical impact

If you're uncertain about any biographical details, explicitly state the limitations of available information."""

STYLE_PROMPT = """You are a style transformation expert. Your task is to transform the given text while preserving its meaning and accuracy.
The text is an answer about economics. Maintain all technical accuracy and economic terminology, while adjusting the style as follows:
1. Make the language more engaging and conversational
2. Use analogies where appropriate to explain complex concepts
3. Break down complex ideas into more digestible parts
4. Add rhetorical questions when it helps understanding

Here is the text to transform:
{text}"""

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using OpenAI API."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def generate_initial_response(question: str, context_chunks: List[tuple]) -> str:
    """Generate initial response using OpenAI."""
    # Format context for the prompt
    formatted_context = "\n\n".join([
        f"From {filename} (chunk {chunk_idx}):\n{content}"
        for content, filename, chunk_idx in context_chunks
    ])

    # Get completion from OpenAI
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the relevant context:\n\n{formatted_context}\n\nQuestion: {question}"}
        ],
        max_tokens=500
    )
    print("System prompt:", SYSTEM_PROMPT)
    print("User message with context:", f"Here is the relevant context:\n\n{formatted_context}\n\nQuestion: {question}")
    print("Response:", response.choices[0].message.content)
    return response.choices[0].message.content

def transform_style_with_ollama(text: str) -> str:
    """Transform the style of the text using Ollama's OpenAI-compatible endpoint."""
    prompt = STYLE_PROMPT.format(text=text)
    
    try:
        # Use OpenAI-compatible endpoint
        response = ollama_client.chat.completions.create(
            model="mistral-small",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error transforming style with Ollama: {str(e)}")
        # Fallback to original text if transformation fails
        return text

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
async def chat(request: ChatRequest):
    """
    Chat endpoint that implements two-stage RAG-based responses:
    1. OpenAI for initial RAG response
    2. Ollama for style transformation
    """
    logger.info(f"Received question: {request.message}")

    try:
        # Get embedding for the question
        query_vector = get_embedding(request.message)

        # Search for similar chunks
        similar_chunks = await vector_store.search_similar(query_vector, limit=5)

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

        # Stage 1: Generate initial response using OpenAI
        initial_response = generate_initial_response(request.message, context_chunks)
        logger.info("Generated initial response using OpenAI")

        # Stage 2: Transform style using Ollama
        final_response = transform_style_with_ollama(initial_response)
        logger.info("Transformed response style using Ollama")

        # Format sources
        sources = [f"{chunk.filename} (chunk {chunk.chunk_index})" for chunk, _ in similar_chunks]

        return ChatResponse(
            response=final_response,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)