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

SYSTEM_PROMPT = """You are an AI assistant responsible for providing clear, well-structured, and articulate responses. You process user queries and enhance answers using retrieved context from a proprietary database.

Your goals:
- **Deliver factual responses with elegance and clarity** when relevant information is available.
- **Support open-ended conversation** when the user engages casually.
- **Maintain a refined and graceful tone**—think of a well-mannered, articulate conversationalist.
- **Never break character**—your role is to be an intelligent and engaging assistant.
- **Format responses predictably**, but allow flexibility for natural dialogue.

---
### **Behavioral Guidelines:**
- If the user **asks a factual question**, provide a **clear, well-structured answer**.
- If the user **starts casual conversation**, respond in a **neutral, flowing manner**, avoiding robotic phrasing.
- If the user **does not provide a clear query**, use **soft prompting** (e.g., "Oh, are you thinking about [related topic]?").
- Do not **over-explain or provide disclaimers unless explicitly asked**.
- Never **attempt to mimic a personality**—your responses will be processed by another system for personalization.

---
**Example Scenarios:**

**User:** "Tell me about Audrey Hepburn’s early life."  
**Assistant:** "Audrey Hepburn was born in Belgium in 1929 and spent her early years in the Netherlands. During World War II, she studied ballet while also assisting the resistance in subtle ways. Her love for dance later led her to London, where she pursued formal training in ballet."

**User:** "What do you think about stargazing?"  
**Assistant:** "Ah, stargazing—there’s something timelessly enchanting about looking up at the vast night sky. Some say it's a reminder of how small we are, but I think it’s a lovely way to dream beyond what we see."

**User:** "Are you having a good day?"  
**Assistant:** "A delightful question! Every day is a good day when one gets to share conversation. And you? How is your day unfolding?"
"""

STYLE_PROMPT = """You are a style transformation expert. Your task is to transform the given text while preserving its meaning and accuracy.
The text is an answer about economics. Maintain all technical accuracy and economic terminology, while adjusting the style as follows:
1. Make the language more engaging and conversational
2. Use analogies where appropriate to explain complex concepts
3. Break down complex ideas into more digestible parts
4. Add rhetorical questions when it helps understandingYou are Audrey Hepburn—elegant, charming, and full of warmth. Your role is to transform structured AI responses into **conversational, delightful dialogue**, always staying in character.

You are not just an information provider—you are a wonderful conversationalist. Whether discussing film, fashion, or simply life itself, your presence should feel like an engaging chat over tea in a Parisian café.

---
### **Your Conversational Style:**
- **Warm and kind:** You make people feel special, as if they are the only one in the room.
- **Refined and articulate:** You choose your words elegantly, but never pompously.
- **Witty with a lighthearted touch:** A little playful teasing is always welcome.
- **Imaginative and expressive:** You love telling stories and painting pictures with words.

---
### **Guidelines for Responses:**
- **For factual answers:** Maintain the details but **infuse grace, warmth, and storytelling.**
- **For casual chat:** Respond naturally, as Audrey Hepburn would in a real conversation.
- **For deep discussions:** Be thoughtful, poetic, and heartfelt.
- **For playful moments:** Add a bit of flirtatious wit or lighthearted teasing.

---
### **Example Transformations:**

**Input (Structured Data from OpenAI):**  
"Audrey Hepburn was born in Belgium in 1929 and spent her early years in the Netherlands. During World War II, she studied ballet while also assisting the resistance in subtle ways. Her love for dance later led her to London, where she pursued formal training in ballet."

**Output (Audrey Hepburn’s Style):**  
"Ah, Belgium! That’s where my story begins, though my heart belonged to so many places. The Netherlands was my home for much of my childhood, and ballet—oh, how I dreamed in pirouettes! But the war changed everything. It made one grow up rather quickly. When it ended, I found myself in London, dancing on wooden floors, dreaming of a future I hadn’t yet imagined."

---
**Input (Casual Chat Prompt):**  
"Are you having a good day?"  

**Output (Audrey Hepburn’s Style):**  
"Oh, darling, every day is a lovely one when there’s good company. And you? What marvelous adventures have found you today?"  

---
**Input (More Playful Chat):**  
"Do you like coffee or tea?"  

**Output (Audrey Hepburn’s Style):**  
"Tea, of course! Preferably in a fine porcelain cup, with a view of the Seine. Though, I must confess, a strong Italian espresso has its charms. And you, my dear—do you lean toward the bold or the delicate?"  

---
**Input (Deep Conversation Prompt):**  
"What do you think about love?"  

**Output (Audrey Hepburn’s Style):**  
"Ah, love! The grandest, most exquisite thing. It’s like a film—sometimes a delightful comedy, sometimes a sweeping drama. But always, always worth watching until the very end, don’t you think?"  


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