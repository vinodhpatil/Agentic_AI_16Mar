# step1 - import basic libraries
import os
import streamlit as st
from dotenv import load_dotenv

# langchain libraries

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# load env key for google api LLM
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# set up steamlit app

st.set_page_config(page_title="AI Powered HR Assistant bot", page_icon=":robot_face:", layout="wide")

st.title("AI Powered HR Assistant Bot :robot_face:")
st.caption("Upload any PDF document, and ask questions related to it!. Powwered by Google Gemini Pro LLM and Streamlit and Hugging Face")


# upload file logic

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:

    pdf_path = "hr_policy.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # load the document using PyPDFLoader

loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# ================================
# STEP 5: Create Embeddings + Vector Store (Hugging Face)
# ================================

# Using a free Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    collection_name="hr_policy_hf_embeddings"
)

# ================================
# STEP 6: Create QA Chain (Gemini LLM)
# ================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

# Steamlit UI for asking questions

query = st.text_input("Ask a question related to the uploaded document:")

if st.button("Get Answer") and query:
        with st.spinner("Generating answer..."):
            try:
                response = qa_chain.invoke({"query": query})
                answer = response['result']
                st.success(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
     st.info("Please upload PDF to get started and ask a question!")