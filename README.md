# Document Query System using RAG with Gemini API

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** that allows users to upload PDF documents and query relevant information. The system retrieves the most relevant chunks from the uploaded documents using **FAISS** for vector storage and **SentenceTransformer** for embeddings. The retrieved text is then passed to **Google's Gemini API**, which generates a well-structured, detailed response along with additional contextual information.

## Objective
The goal of this project is to:
- Efficiently extract and preprocess text from PDFs.
- Index and retrieve relevant document sections using FAISS.
- Enhance responses using Google's Gemini API.
- Present results in an interactive Streamlit interface with highlights and structured formatting.

## Workflow
1. **Upload PDF**: Users upload a PDF document.
2. **Text Extraction**: The system extracts text from the PDF using `PyPDF2`.
3. **Preprocessing & Chunking**:
   - Text is cleaned to remove unwanted characters.
   - It is split into smaller chunks using `RecursiveCharacterTextSplitter`.
4. **Vector Embedding & Storage**:
   - Each chunk is converted into embeddings using `multi-qa-MiniLM-L6-cos-v1` from `SentenceTransformer`.
   - FAISS indexes these embeddings for fast retrieval.
5. **Query Processing**:
   - The user enters a query.
   - The system retrieves the most relevant chunks from FAISS.
   - The retrieved chunks are highlighted and structured.
6. **Gemini API Response**:
   - The relevant text is passed to `Gemini-pro`.
   - The API generates a detailed response, including additional context.
7. **Display Results**:
   - The response is displayed in a user-friendly format on Streamlit.
   - Important words are highlighted for better readability.

## Tech Stack
- **Frontend:** Streamlit (UI for document upload and query interface)
- **Backend:** Python
- **Libraries Used:**
  - `streamlit`: Web application framework
  - `faiss-cpu`: Fast indexing and retrieval
  - `numpy`: Numerical operations
  - `PyPDF2`: PDF text extraction
  - `sentence-transformers`: Embeddings for document search
  - `langchain`: Text chunking and processing
  - `summa`: TextRank summarization
  - `google-generativeai`: Interface for Gemini API
- **API Used:** Google Gemini API for response generation

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-query-system.git
   cd document-query-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API key:
   - Create a `.streamlit/secrets.toml` file.
   - Add the following content:
     ```toml
     api_key = "your_gemini_api_key"
     ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Upload a PDF document.
3. Enter a query in the search bar.
4. View the retrieved and AI-enhanced response.

## Future Improvements
- **Multi-PDF support** for querying across multiple documents.
- **Enhanced UI** with response formatting and interactive elements.
- **Metadata storage** to associate documents with users.
- **Citations and sources** for more credible responses.

---
This project provides an efficient document query system that combines RAG with LLM capabilities, making information retrieval more accurate and context-aware.

