import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag import get_retriever
from config import MODEL_NAME


# -----------------------------
# Load LLM
# -----------------------------

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=1024
)

# -----------------------------
# Format Retrieved Documents
# -----------------------------

def format_docs(docs):
    context = ""

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")

        context += f"""

Source: {source}

{doc.page_content}

"""

    return context


# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_template("""

You are the official AI HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.

Answer ONLY using the HR policy documents provided.

Rules:

1. Never make up information.

2. If the answer is not found in the documents, reply exactly:

"I couldn't find this information in the Zyro Dynamics HR policy documents."

3. If the question is unrelated to HR policies, reply:

"I can only answer questions related to Zyro Dynamics HR policy documents."

4. Keep answers concise.

5. Mention policy details whenever possible.

6. Mention source document names.

Context:

{context}

Question:

{question}

Answer:

""")


# -----------------------------
# Retriever
# -----------------------------

retriever = get_retriever()


# -----------------------------
# RAG Chain
# -----------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# HR Keywords
# -----------------------------

HR_KEYWORDS = [
    "leave",
    "salary",
    "benefits",
    "employee",
    "travel",
    "expense",
    "probation",
    "performance",
    "promotion",
    "bonus",
    "policy",
    "attendance",
    "joining",
    "termination",
    "maternity",
    "paternity",
    "wfh",
    "remote",
    "security",
    "posh"
]


# -----------------------------
# Check HR Question
# -----------------------------

def is_hr_question(question):

    question = question.lower()

    return any(
        keyword in question
        for keyword in HR_KEYWORDS
    )


# -----------------------------
# Chatbot Function
# -----------------------------

def ask_hr_bot(question):

    if not is_hr_question(question):

        return (
            "I can only answer questions related to "
            "Zyro Dynamics HR policy documents."
        )

    return rag_chain.invoke(question)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Zyro Dynamics HR Help Desk")

st.write("Ask any HR-related question.")


# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# Chat Input
# -----------------------------

query = st.chat_input("Ask your HR question...")


# -----------------------------
# Generate Response
# -----------------------------

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching HR Policies..."):

            response = ask_hr_bot(query)

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )