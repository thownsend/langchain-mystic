import os
import uuid
import chromadb
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.url_utils import is_valid_url
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile,gettempdir

load_dotenv()

azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
openai_api_version = os.getenv("OPENAI_API_VERSION")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=openai_api_version,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
)
st.set_page_config(page_title="Veri Çekme", page_icon="📈")

st.markdown("# Veri Çekme")

if "link_listesi" not in st.session_state:
    st.session_state.link_listesi = []

persistent_client = chromadb.PersistentClient(path="./vectors/lugado")

db = Chroma(
    client=persistent_client,
    collection_name="lugado_website_contents",
    embedding_function=embeddings,
)


# Eğer kullanıcı bir link girdiyse, listeye ekle ve göster
link = st.text_input("Linki girin:")


def verileri_cek():
    docs = []
    with st.status("Downloading data..."):
        st.write("Searching for data...")

        loader = WebBaseLoader(st.session_state.link_listesi)
        loader.requests_kwargs = {"verify": False}
        docs = loader.load()

        documents = text_splitter.split_documents(docs)

        st.write("Embeddings saved.")

        st.write(documents)
        db.add_documents(documents)
        st.session_state.link_listesi = []


uploaded_file = st.file_uploader("PDF Seç", type="pdf")
if uploaded_file is not None:
    temp_dir = gettempdir()
    with NamedTemporaryFile(dir=temp_dir, suffix=".pdf", delete=False) as f:
        f.write(uploaded_file.getbuffer())
        temp_file_path = f.name
    
    try:
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        documents = text_splitter.split_documents(pages)

        st.write(documents)
        db.add_documents(documents)
        st.write("PDF Embeddings saved.")
    finally:
        os.remove(temp_file_path)


button2 = st.button("Verileri Çek", key="fetch", on_click=verileri_cek)
button = st.button("Ekle", key="ekle")
if button:
    if link != "" and is_valid_url(link):
        if link in st.session_state.link_listesi:
            st.error("Link zaten var!")
        else:
            st.session_state.link_listesi.append(link)
            st.success("URL başarıyla eklendi!", icon="✅")
    else:
        st.error("URL hatalı!")
if st.session_state.link_listesi and len(st.session_state.link_listesi) > 0:
    st.header("Linkler:")
    for i, link in enumerate(st.session_state.link_listesi, start=1):
        st.write(f"{i}. {link}")
else:
    st.info("Henüz ekli bir link yok")