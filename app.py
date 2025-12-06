import streamlit as st
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
DOC_ID = '1fM-m6htzgFhw5okJtoroIBVZTHeclO410OdSQryrETg' # Extracted from your link
DOC_LINK = "https://docs.google.com/document/d/1fM-m6htzgFhw5okJtoroIBVZTHeclO410OdSQryrETg/edit?tab=t.0#heading=h.t3i2a5k1kyhq"

st.set_page_config(page_title="Live Policy Bot")
st.title("🤖 Live Google Doc Assistant")

# --- SIDEBAR: SECRETS ---
# We look for secrets in Streamlit Cloud or local environment
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- GOOGLE DOCS FETCHER ---
def get_google_doc_content():
    """Fetches text from the live Google Doc."""
    try:
        # Load credentials from Streamlit secrets
        gcp_service_account = json.loads(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            gcp_service_account, scopes=['https://www.googleapis.com/auth/documents.readonly']
        )
        service = build('docs', 'v1', credentials=creds)
        document = service.documents().get(documentId=DOC_ID).execute()
        
        # Extract text from the JSON structure
        text = ""
        for content in document.get('body').get('content'):
            if 'paragraph' in content:
                elements = content.get('paragraph').get('elements')
                for elem in elements:
                    if 'textRun' in elem:
                        text += elem.get('textRun').get('content')
        return text
    except Exception as e:
        st.error(f"Error accessing Google Doc: {e}")
        return None

# --- KNOWLEDGE BASE SETUP ---
def setup_knowledge_base(text_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text_content)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# --- LOAD DATA BUTTON ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.button("🔄 Refresh Data from Google Doc"):
    with st.spinner("Fetching latest updates from Google Docs..."):
        if "gcp_service_account" not in st.secrets:
            st.error("Google Credentials not found in Secrets!")
        else:
            raw_text = get_google_doc_content()
            if raw_text:
                st.session_state.vectorstore = setup_knowledge_base(raw_text)
                st.success("Knowledge base updated!")

# --- CHAT ENGINE ---
if st.session_state.vectorstore:
    # Your Specific Rules
    custom_template = f"""You are a helpful assistant based on a Google Document.
    
    STRICT RULES:
    1. If the answer is not in the context provided, reply EXACTLY: "Please find your RO for clarification."
    2. If the user asks for a visual, diagram, or clearer illustration, reply EXACTLY: "Please refer to the same google docs: {DOC_LINK}"
    3. Keep answers professional.

    Context:
    {{context}}

    Question: {{question}}
    Answer:"""
    
    PROMPT = PromptTemplate(template=custom_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = qa_chain.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please click 'Refresh Data' to load the document initially.")