import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import faiss
import numpy as np
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def pdf_read(file_path):
    text = ""
    pdf_read = PdfReader(file_path)
    for page in pdf_read.pages:
        text += page.extract_text()
    return text

def get_chunk(text):
    split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = split.split_text(text)
    return chunks

def query_faiss_index(query, embeddings, index, chunks, top_k=3):
    
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")
    
    
    distances, indices = index.search(query_embedding, top_k)
    
   
    closest_chunks = [chunks[i] for i in indices[0]]
    return closest_chunks

st.title("PDF Search with Google Generative AI and FAISS")


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    
    st.write("Processing PDF file...")
    text = pdf_read(uploaded_file)
    chunks = get_chunk(text)
    
    
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("Google API Key not found. Please check your .env file.")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
       
        chunk_embeddings = []
        for chunk in chunks:
            embedding = embeddings.embed_query(chunk)
            chunk_embeddings.append(embedding)
        
        
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")
        dimension = chunk_embeddings.shape[1]  
        index = faiss.IndexFlatL2(dimension)  
        
        
        index.add(chunk_embeddings)
        
        
        query = st.text_input("Enter a query for search:")
        
        if query:
            st.write(f"Searching for: {query}")
            result_chunks = query_faiss_index(query, embeddings, index, chunks)
            
           
            st.write("Closest chunks to the query:")
            for i, chunk in enumerate(result_chunks):
                st.write(f"**Chunk {i+1}:**\n{chunk}\n")

