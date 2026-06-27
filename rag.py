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
    CHUNK_OVERLAP,
)

# ---------------------------------------------------
# Load all PDFs
# ---------------------------------------------------

def load_documents():

    documents = []

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data folder not found: {DATA_PATH}")

    pdf_files = sorted(
        [
            file
            for file in os.listdir(DATA_PATH)
            if file.endswith(".pdf")
        ]
    )

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf in pdf_files:

        path = os.path.join(DATA_PATH, pdf)

        loader = PyPDFLoader(path)

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf

        documents.extend(docs)

    print(f"Loaded {len(documents)} pages.")

    return documents


# ---------------------------------------------------
# Split Documents
# ---------------------------------------------------

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

    print(f"Created {len(chunks)} chunks.")

    return chunks


# ---------------------------------------------------
# Embedding Model
# ---------------------------------------------------

def get_embeddings():

    return HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }

    )


# ---------------------------------------------------
# Create VectorStore
# ---------------------------------------------------

def create_vectorstore():

    print("Creating FAISS Index...")

    docs = load_documents()

    chunks = split_documents(docs)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(

        chunks,

        embeddings

    )

    os.makedirs(FAISS_PATH, exist_ok=True)

    vectorstore.save_local(FAISS_PATH)

    print("FAISS Index Saved.")

    return vectorstore


# ---------------------------------------------------
# Load VectorStore
# ---------------------------------------------------

def load_vectorstore():

    embeddings = get_embeddings()

    return FAISS.load_local(

        FAISS_PATH,

        embeddings,

        allow_dangerous_deserialization=True

    )


# ---------------------------------------------------
# Initialize VectorStore
# ---------------------------------------------------

def initialize_vectorstore():

    index_file = os.path.join(
        FAISS_PATH,
        "index.faiss"
    )

    pkl_file = os.path.join(
        FAISS_PATH,
        "index.pkl"
    )

    if os.path.exists(index_file) and os.path.exists(pkl_file):

        print("Loading Existing FAISS Index...")

        return load_vectorstore()

    print("No FAISS Index Found.")

    return create_vectorstore()


# ---------------------------------------------------
# Retriever
# ---------------------------------------------------

# ---------------------------------------------------
# Get Retriever
# ---------------------------------------------------

def get_retriever():

    vectorstore = initialize_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 12,
            "lambda_mult": 0.75
        }
    )

    return retriever