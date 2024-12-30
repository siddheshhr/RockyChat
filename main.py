import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from transformers import pipeline

# Load Hugging Face API Token from environment
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Streamlit setup
st.title("RockyBot: News Research Tool with Hugging Face API 📈")
st.sidebar.title("News Article URLs")

# Input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"
main_placeholder = st.empty()

# Initialize Hugging Face pipeline for question-answering
qa_pipeline = pipeline("question-answering", model="meta-llama/Llama-3.1-8B", use_auth_token=huggingface_token)

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started ✅")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting... Started ✅")
    docs = text_splitter.split_documents(data)

    # Save document chunks locally for future use
    with open(file_path, "wb") as f:
        pickle.dump(docs, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            docs = pickle.load(f)

            # Combine the document chunks into a single context
            context = "\n".join([doc.page_content for doc in docs])

            # Use Hugging Face QA pipeline
            main_placeholder.text("Processing query with Hugging Face API... ✅")
            result = qa_pipeline(question=query, context=context)

            # Display results
            st.header("Answer")
            st.write(result.get("answer", "No answer found."))

            # Display confidence score
            st.subheader("Confidence Score")
            st.write(result.get("score", "N/A"))
