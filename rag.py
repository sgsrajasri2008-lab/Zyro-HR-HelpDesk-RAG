import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    DATA_PATH,
    FAISS_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# -----------------------------
# Load all PDF documents
# -----------------------------
def load_documents():
    documents = []

    pdf_files = sorted([
        file for file in os.listdir(DATA_PATH)
        if file.endswith(".pdf")
    ])

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    for file in pdf_files:

        file_path = os.path.join(DATA_PATH, file)

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)

    print(f"Loaded {len(documents)} pages.\n")

    return documents


# -----------------------------
# Split documents
# -----------------------------
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.\n")

    return chunks


# -----------------------------
# Embedding Model
# -----------------------------
def get_embedding_model():

    return HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }

    )


# -----------------------------
# Build FAISS Index
# -----------------------------
def create_vectorstore(chunks):

    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(

        chunks,

        embeddings

    )

    vectorstore.save_local(FAISS_PATH)

    print("FAISS Index Saved Successfully.\n")

    return vectorstore


# -----------------------------
# Load Existing FAISS Index
# -----------------------------
def load_vectorstore():

    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(

        FAISS_PATH,

        embeddings,

        allow_dangerous_deserialization=True

    )

    print("FAISS Index Loaded Successfully.\n")

    return vectorstore


# -----------------------------
# Get Retriever
# -----------------------------
def get_retriever():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={

            "k": 5,

            "fetch_k": 12,

            "lambda_mult": 0.75

        }

    )

    return retriever


# -----------------------------
# Build Vector Database
# -----------------------------
def initialize_vectorstore():

    if os.path.exists(FAISS_PATH):

        print("Existing FAISS Index Found.\n")

        return load_vectorstore()

    print("Creating New FAISS Index...\n")

    documents = load_documents()

    chunks = split_documents(documents)

    return create_vectorstore(chunks)


# -----------------------------
# Test File
# -----------------------------
if __name__ == "__main__":

    initialize_vectorstore()