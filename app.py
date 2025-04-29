import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
import re
import spacy
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from summa import summarizer  
import os
from dotenv import load_dotenv



load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


import importlib.util

# Auto-download en_core_web_sm if not present
if not importlib.util.find_spec("en_core_web_sm"):
    from spacy.cli import download
    download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")


embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def pdf_read(file_path):
    """Extracts text from PDF file."""
    text = ""
    pdf_reader = PdfReader(file_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  
    return text

def clean_text(text):
    """Removes noise, unwanted characters, and fixes grammar."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def summarize_text(text):
    """Summarizes text using TextRank."""
    return summarizer.summarize(text, ratio=0.3)

def get_chunk(text, chunk_size=500, chunk_overlap=100):
    """Splits text into chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def highlight_text(text, query):
    """Highlights the matched words in the result and formats with bullet points."""
    words = query.split()
    for word in words:
        text = re.sub(f"(?i)({word})", r"**\1**", text)  

    sentences = re.split(r'(?<=[.!?])\s+', text)  
    formatted_text = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])
    
    return formatted_text

def query_faiss_index(query, model, index, chunks, top_k=5):
    """Searches FAISS index for the most relevant chunks."""
    if not chunks:
        return []  # Return an empty list if no chunks exist
    
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append((chunks[idx], dist))
    return results

def generate_response(query, results):
    """Generates response using Gemini API."""
    if not results:
        return "⚠️ No relevant content found in the PDF. Please try a different query."
    
    context = "\n".join([highlight_text(result[0], query) for result in results])
    prompt = f"### Instruction: Answer the following question based on the given context.also explain it in detail form based on that content.but the extra content should be shown separately like additional information about the query.\n\nQuestion: {query}\nContext:\n{context}\n\n### Response:"
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    try:
        _result = response._result  
        
        if hasattr(_result, 'candidates') and _result.candidates:
            candidate = _result.candidates[0]  
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text  
        return "⚠️ No valid candidates found."
    except AttributeError as e:
        return f"⚠️ AttributeError: {e}"
    except Exception as e:
        return f"⚠️ An error occurred: {e}"

st.set_page_config(page_title="AI RAG PDF Assistant", page_icon="🔍", layout="wide")

st.markdown("""
    <h2 style="text-align: center; color: #4CAF50;">📘 AI RAG PDF Assistant</h2>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="border: 2px solid #2196F3; padding: 10px; border-radius: 10px; background-color: #f0f8ff;">
        <h4 style="color: #333; text-align: center;">🔍 About This Assistant</h4>
        <p style="font-size: 16px; color: #000; text-align: justify;">
        This application allows users to upload PDF documents and ask questions about their content. 
        It is suitable for normal books, research articles, academic papers, and any text-based PDFs.
        </p>
    </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Choose a PDF file ", type="pdf")

if uploaded_file is not None:
    st.info("📄 Processing PDF file... Please wait.", icon="🔄")
    
    raw_text = pdf_read(uploaded_file)
    cleaned_text = clean_text(raw_text)
    summarized_text = summarize_text(cleaned_text)
    chunks = get_chunk(summarized_text)
    
    if chunks:
        chunk_embeddings = embedding_model.encode(chunks)
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)
    else:
        index = None  # No chunks, no FAISS index

    query = st.text_input("💡 Ask a question related to the PDF:")

    if query:
        st.write(f"🔍 Searching for: **{query}**")
        results = query_faiss_index(query.lower(), embedding_model, index, chunks) if index else []

        response = generate_response(query, results)
        st.markdown(f"<div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;'><h4 style='color: #333;'>📝 Generated Answer:</h4><p style='font-size: 16px; color: #000;'>{response}</p></div>", unsafe_allow_html=True)

        if results:
            st.write("📌 **Top matching results from the PDF:**")
            for i, (chunk, score) in enumerate(results):
                st.write(f"{i+1}. {highlight_text(chunk, query)}")
        else:
            st.warning("⚠️ No relevant content found in the PDF.")
