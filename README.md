# Economics Expert Chat 📚

A RAG-powered chatbot that answers questions about economics using content from "Principles of Economics" textbook. The system uses OpenAI's embeddings for semantic search and GPT-4 for generating accurate, context-aware responses.

> **Note**: This entire project was implemented using Cursor's AI Agent mode, serving as a demonstration of Cursor's capabilities. No manual coding or editing was performed - all code was generated and modified through AI agent interactions.

## Features

- 📖 Semantic search over economics textbook content
- 🤖 GPT-4 powered responses with source attribution
- 🔍 Retrieval-Augmented Generation (RAG) for accurate answers
- 💻 Modern web interface built with Streamlit
- 🚀 Fast and scalable FastAPI backend
- 🧪 Complete AI-assisted development using Cursor

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd econ-bot
```

2. Download the economics textbook:
   - Visit [Principles of Economics LibreTexts](https://socialsci.libretexts.org/Bookshelves/Economics/Principles_of_Economics_(LibreTexts))
   - Click on "Download Full Book (PDF)" in the left sidebar
   - Save the file as `principles_of_economics.pdf` in the project root directory

3. Create and activate a virtual environment:
```bash
python -m venv econ_venv
source econ_venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Components

### 1. Document Ingestion

The ingestion script processes the PDF textbook and stores its contents in the vector database.

To run the ingestion process:
```bash
python ingest.py
```

This will:
- Extract text from the PDF
- Split it into manageable chunks
- Generate embeddings using OpenAI's API
- Store the chunks and embeddings in the database

### 2. Backend Server

The FastAPI backend handles chat requests and implements the RAG system.

To start the backend server:
```bash
python app.py
```

The server will start at `http://localhost:8000` with the following endpoints:
- POST `/chat`: Main chat endpoint that accepts questions and returns AI-generated responses

### 3. Frontend Interface

The Streamlit frontend provides a user-friendly chat interface.

To start the frontend:
```bash
streamlit run streamlit_app.py
```

This will open the chat interface in your default web browser at `http://localhost:8501`.

## System Architecture

1. **Document Processing**:
   - PDF text extraction using PyPDF
   - Text chunking with tiktoken
   - Embedding generation with OpenAI's text-embedding-3-small model

2. **Storage**:
   - Vector storage using Qdrant
   - SQLite database for metadata

3. **Query Processing**:
   - Question embedding generation
   - Semantic similarity search
   - Context-aware response generation with GPT-4

## Dependencies

- Python 3.8+
- OpenAI API (for embeddings and chat completion)
- FastAPI (backend server)
- Streamlit (frontend interface)
- Qdrant (vector database)
- PyPDF (PDF processing)
- SQLAlchemy (database ORM)

## Development

To contribute or modify:

1. The main components are:
   - `ingest.py`: Document processing and database population
   - `app.py`: FastAPI backend implementation
   - `streamlit_app.py`: Streamlit frontend
   - `database.py`: Database and vector store operations

2. Key configuration files:
   - `requirements.txt`: Python dependencies

## Notes

- Ensure all components (backend and frontend) are running simultaneously for the system to work
- The system requires an active internet connection for OpenAI API calls
- Monitor your OpenAI API usage as both embeddings and chat completions consume tokens