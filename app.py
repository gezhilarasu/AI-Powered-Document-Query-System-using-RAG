import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
import re

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from summa import summarizer  # For TextRank Summarization

# Set up Gemini API Key
GEMINI_API_KEY = "AIzaSyBUlbog2X1_lk1cLGzcb0z7QQ5Wc_Ft3ew"
genai.configure(api_key=st.secrets["api_key"])

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def pdf_read(file_path):
    """Extracts text from PDF file."""
    text = ""
    pdf_reader = PdfReader(file_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle pages with no extractable text
    return text

def clean_text(text):
    """Removes noise, unwanted characters, and fixes grammar."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive whitespace
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
        text = re.sub(f"(?i)({word})", r"**\1**", text)  # Case-insensitive highlight

    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by punctuation followed by space
    formatted_text = "\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])
    
    return formatted_text

def query_faiss_index(query, model, index, chunks, top_k=5):
    """Searches FAISS index for the most relevant chunks."""
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append((chunks[idx], dist))
    return results

import json

def generate_response(query, results):
    """Generates response using Gemini API."""
    context = "\n".join([highlight_text(result[0], query) for result in results])
    prompt = f"### Instruction: Answer the following question based on the given context.also explain it in detail form based on that content.but the extra content should be shown separately like additional information about the query.\n\nQuestion: {query}\nContext:\n{context}\n\n### Response:"

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)



    # Step 3: Try to access the _result attribute instead of result
    try:
        _result = response._result  # Try accessing the _result attribute
        

        if hasattr(_result, 'candidates') and _result.candidates:
           

            candidate = _result.candidates[0]  # Access the first candidate
            if hasattr(candidate, 'content') and candidate.content:
                
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    
                    return candidate.content.parts[0].text  # Extract the text

        return "⚠️ No valid candidates found."
    
    except AttributeError as e:
        st.write(f"⚠️ AttributeError: {e}")
        return f"⚠️ AttributeError: {e}"

    except Exception as e:
        st.write(f"⚠️ An error occurred: {e}")
        return f"⚠️ An error occurred: {e}"


# Streamlit layout
st.set_page_config(page_title="RAG AI ASSISTANT", page_icon="🔍", layout="wide")
st.title("📘 AI RAG PDF Assistant")
# File upload
uploaded_file = st.file_uploader("📂 Choose a PDF file to summarize", type="pdf")

if uploaded_file is not None:
    st.info("📄 Processing PDF file... Please wait.", icon="🔄")
    
    raw_text = pdf_read(uploaded_file)
    cleaned_text = clean_text(raw_text)
    summarized_text = summarize_text(cleaned_text)
    chunks = get_chunk(summarized_text)
    
    chunk_embeddings = embedding_model.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)

    query = st.text_input("💡 Ask a question related to the PDF:")

    if query:
        st.write(f"🔍 Searching for: **{query}**")
        results = query_faiss_index(query.lower(), embedding_model, index, chunks)

        response = generate_response(query, results)
        st.write("📝 **Generated Answer:**")
        st.markdown(response)

        st.write("📌 **Top matching results:**")
        for i, (chunk, score) in enumerate(results):
            st.write(f"{i+1}. {highlight_text(chunk, query)}")

        