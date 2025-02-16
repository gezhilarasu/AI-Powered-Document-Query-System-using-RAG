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

