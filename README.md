<<<<<<< HEAD
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
=======
# 📄 DocQueryAI: Intelligent Document Query System  

## 🌟 Overview  
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and query relevant information. The system retrieves the most relevant chunks from the uploaded documents using FAISS for vector storage and SentenceTransformer for embeddings. The retrieved text is then passed to Google's Gemini API, which generates a well-structured, detailed response along with additional contextual information.  

---

## 🎯 Objective  
- 📝 **Extract text** from PDFs efficiently.  
- 🔍 **Retrieve relevant answers** using FAISS for similarity search.  
- 🤖 **Enhance responses** with Gemini AI to provide detailed and structured answers.  
- 🎨 **Highlight key answers** for better readability.  

---

## 🔄 Workflow  

1️⃣ **📂 Upload PDF** → User uploads a document.  
2️⃣ **📖 Extract Text** → Extracts text using `PyPDF2`.  
3️⃣ **🧹 Clean & Preprocess** → Removes noise and structures the content.  
4️⃣ **📌 Split into Chunks** → Uses `RecursiveCharacterTextSplitter` for better retrieval.  
5️⃣ **📊 Store in FAISS** → Converts text into embeddings using `SentenceTransformer` and indexes it in FAISS.  
6️⃣ **🔍 User Query** → Searches the indexed database for the most relevant chunk.  
7️⃣ **🤖 Generate Response** → Passes the result to `Gemini AI` for a detailed answer with additional information.  
8️⃣ **🌟 Display Answer** → Highlights relevant text and presents structured responses.  

---

## 🛠️ Tech Stack  

| Tech  | Usage  |
|-------|--------|
| 🐍 **Python**  | Core language for development  |
| 🎨 **Streamlit**  | Web UI for user interaction  |
| 📄 **PyPDF2**  | PDF text extraction  |
| 🧠 **FAISS**  | Vector storage & retrieval  |
| 🤖 **SentenceTransformer**  | Embeddings for document search  |
| 📝 **LangChain**  | Text chunking & retrieval  |
| ⚡ **Gemini AI API**  | Generates enhanced responses  |
| 🔍 **Responsive answer**  | Final answer  |

---

## 📌 Usage  
1️⃣ **Upload a PDF** 📂  
2️⃣ **Enter a query** 🔍  
3️⃣ **View AI-enhanced results** 🤖✨  

---

## 🥁Future Improvements

**Multi-PDF support for querying across multiple documents.**

**Enhanced UI with response formatting and interactive elements.**

**Metadata storage to associate documents with users.**

## 🤝 Contributing  
Feel free to **fork** the repository and submit **pull requests**! 🔥  

---
Happy Coding! 🚀🎉
>>>>>>> d1695c3b0e74d000569fc8baec1fb3a569fc4588

