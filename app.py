## RAG Q&A Conversation with PDF Including Chat History

import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# -------------------------
# Streamlit Secrets (Cloud-safe)
# -----------------------
required_secrets = ["HF_TOKEN", "GROQ_API_KEY"]
for key in required_secrets:
    if key not in st.secrets:
        st.error(f"{key} is missing. Please add it in Streamlit Secrets.")
        st.stop()

os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# -----------------------
# Streamlit UI
# -----------------------
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content.")

session_id = st.text_input("Session ID:", value="default_session")


# -----------------------
# LLM (Groq)
# -----------------------
llm = ChatGroq(model="groq/compound-mini")


# -----------------------
# Embeddings (HuggingFace)
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------
# Chat History Store
# -----------------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# -----------------------
# PDF Upload + Vector Store
# -----------------------
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        temp_pdf = f"./temp_{uploaded_file.name}"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        documents.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    splits = text_splitter.split_documents(documents)

    # Vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    st.success("✅ Vector store created!")


# -----------------------
# Conversational RAG
# -----------------------
user_input = st.text_input("Your question:")

if user_input and uploaded_files:

    session_history = get_session_history(session_id)

    retrieved_docs = retriever.invoke(user_input)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_prompt = """
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't know".
Use maximum 3 sentences.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    formatted_prompt = prompt.format_prompt(
        context=context_text,
        input=user_input,
        chat_history=session_history.messages
    )

    start_time = time.process_time()
    response = llm.invoke(formatted_prompt.to_messages())
    elapsed = time.process_time() - start_time

    answer = response.content

    session_history.add_user_message(user_input)
    session_history.add_ai_message(answer)

    st.write(f"⏱ Response time: {elapsed:.2f} sec")
    st.write("### Assistant Answer:")
    st.write(answer)

    st.write("### Chat History:")
    for msg in session_history.messages:
        st.write(f"**{msg.type.capitalize()}**: {msg.content}")

    with st.expander("📄 Retrieved Documents (top 5)"):
        for i, doc in enumerate(retrieved_docs):
            st.write(f"--- Document {i+1} ---")
            st.write(doc.page_content)

