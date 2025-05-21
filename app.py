import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# App layout
st.set_page_config(page_title="📘 Ask Your PDF with Gemini", layout="wide")
st.title("📘 Ask Questions from Your PDF using Gemini Pro")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Reading and processing your PDF..."):
        # Save the uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        # Set up Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        # Create FAISS vector store
        texts = [doc.page_content for doc in split_docs]
        vectorstore = FAISS.from_texts(texts, embedding=embeddings)

        # Set up retriever and Gemini LLM
        retriever = vectorstore.as_retriever()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success("✅ PDF processed successfully! Ask away ⬇️")

        # Ask questions
        query = st.text_input("💬 Ask a question about your PDF:")

        if st.button("Answer"):
            if query:
                with st.spinner("🤖 Thinking..."):
                    result = qa_chain.run(query)
                    st.markdown("### 📍 Answer:")
                    st.write(result)
            else:
                st.warning("Please enter a question first.")
