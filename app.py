import os
import re
import time
import random
import pickle
import streamlit as st
from typing import List
from rouge_score import rouge_scorer

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves the chat message history for a given session ID."""
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

def extract_text_from_pdfs(directory: str) -> List[str]:
    """Extracts text from PDF files in the specified directory."""
    extracted_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            try:
                loader = PyPDFLoader(file_path=pdf_path)
                st.write(f"Processing: {filename}")
                extracted_docs.extend(loader.load_and_split())
            except Exception as e:
                st.error(f"Error processing {filename}: {e}")
    return extracted_docs

def split_text_chunks(documents: List[str]) -> List[str]:
    """Splits documents into smaller text chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def create_vector_database(chunks: List[str], directory: str) -> Chroma:
    """Creates a vector database from document chunks."""
    st.write("Generating vector database...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ["API_KEY"]),
        persist_directory=os.path.join(directory, "vectordb"),
    )

def conversational_rag_pipeline(vectordb: Chroma) -> RunnableWithMessageHistory:
    """Creates a conversational RAG pipeline with history-aware retrieval."""
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer the question concisely."
        " If unsure, say 'I don't know'.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=os.environ["API_KEY"])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))

def compute_rouge_l(reference: str, generated: str) -> float:
    """Calculates ROUGE-L score for answer relevance."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, generated)["rougeL"].fmeasure

def display_answer(answer: str, references: List[str], confidence: float):
    """Displays the chatbot's answer along with confidence score and references."""
    st.markdown(f"### 🤖 Answer (Confidence: {confidence*100:.2f}%)")
    with st.chat_message("assistant"):
        st.write_stream(stream(answer))
    
    st.markdown("#### 📌 References:")
    for ref in references:
        st.markdown(f"- {ref.page_content}")

def main():
    """Main function to run the chatbot UI."""
    st.set_page_config(page_title="📜 Chatbot: Fun Financial Insights!", page_icon="💰")
    st.markdown("<h2 style='text-align: center; color: #F39C12;'>💡 FunBot: Ask Financial Questions! 💡</h2>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.session_state["session_id"] = str(random.getrandbits(32))
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    with st.sidebar:
        st.markdown("### 📂 Upload PDFs")
        uploaded_files = st.file_uploader("Upload financial reports:", accept_multiple_files=True)
        
        if uploaded_files and st.button("Process PDFs"):
            os.makedirs("data/pdfs", exist_ok=True)
            for file in uploaded_files:
                with open(f"data/pdfs/{file.name}", "wb") as f:
                    f.write(file.getbuffer())
            st.success("📄 PDFs uploaded successfully!")
            
            with st.spinner("Extracting and indexing..."):
                docs = extract_text_from_pdfs("data/pdfs/")
                text_chunks = split_text_chunks(docs)
                vectordb = create_vector_database(text_chunks, "data")
                st.session_state["vectordb"] = vectordb
                st.success("✅ PDFs processed!")
    
    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        if "rag_chain" not in st.session_state:
            vectordb = st.session_state.get("vectordb")
            if not vectordb:
                st.error("Please process PDFs before asking questions.")
                return
            st.session_state["rag_chain"] = conversational_rag_pipeline(vectordb)
        
        with st.spinner("Thinking..."):
            response = st.session_state["rag_chain"].invoke({"input": user_input}, config={"configurable": {"session_id": st.session_state["session_id"]}})
            answer = response["answer"]
            references = response.get("context", [])
            rouge_l_score = compute_rouge_l(" ".join([ref.page_content for ref in references]), answer)
            display_answer(answer, references, rouge_l_score)

if __name__ == "__main__":
    main()
