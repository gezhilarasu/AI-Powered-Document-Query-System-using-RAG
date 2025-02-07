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
from io import BytesIO

# Load API Key from Streamlit Secrets
genai.configure(api_key=st.secrets["api_key"])

# Load NLP model and Embedding Model
nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def pdf_read(file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text):
    """Removes noise and unwanted characters."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def summarize_text(text):
    """Summarizes text using TextRank."""
    return summarizer.summarize(text, ratio=0.3)

def get_chunk(text, chunk_size=500, chunk_overlap=100):
    """Splits text into chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def highlight_text(text, query):
    """Highlights matched words in results."""
    words = query.split()
    for word in words:
        text = re.sub(f"(?i)({word})", r"**\1**", text)
    sentences = re.split(r'(?<=[.!?])\s+', text)  
    return "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])

def query_faiss_index(query, model, index, chunks, top_k=5):
    """Searches FAISS index for most relevant chunks."""
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [(chunks[idx], dist) for idx, dist in zip(indices[0], distances[0])]

def generate_response(query, results):
    """Generates a response using Gemini API."""
    context = "\n".join([highlight_text(result[0], query) for result in results])
    prompt = f"""
    ### Instruction: Answer the question based on the given context. Explain in detail, providing additional information separately.
    
    Question: {query}
    Context:
    {context}
    
    ### Response:
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text if response.text else "⚠️ No valid response found."

# Streamlit App UI
st.set_page_config(page_title="AI RAG PDF Assistant", page_icon="🔍", layout="wide")
st.markdown("""<h2 style="text-align: center; color: #4CAF50;">📘 AI RAG PDF Assistant</h2>""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Choose a PDF file", type="pdf")

if uploaded_file:
    st.info("📄 Processing PDF file... Please wait.", icon="🔄")
    raw_text = pdf_read(BytesIO(uploaded_file.read()))
    cleaned_text = clean_text(raw_text)
    summarized_text = summarize_text(cleaned_text)
    chunks = get_chunk(summarized_text)
    
    # Create FAISS index if not already done
    if "faiss_index" not in st.session_state:
        chunk_embeddings = embedding_model.encode(chunks).astype("float32")
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)
        st.session_state.faiss_index = index
        st.session_state.chunks = chunks
    
    query = st.text_input("💡 Ask a question related to the PDF:")
    if query:
        st.write(f"🔍 Searching for: **{query}**")
        results = query_faiss_index(query.lower(), embedding_model, st.session_state.faiss_index, st.session_state.chunks)
        response = generate_response(query, results)
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9;">
            <h4 style="color: #333;">📝 Generated Answer:</h4>
            <p style="font-size: 16px; color: #000;">{response}</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("📌 **Top matching results from the PDF:**")
        for i, (chunk, _) in enumerate(results):
            st.write(f"{i+1}. {highlight_text(chunk, query)}")
