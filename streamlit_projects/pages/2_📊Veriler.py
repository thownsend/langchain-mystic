import chromadb
import pandas as pd
import streamlit as st
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from dotenv import load_dotenv
import os

load_dotenv()
host = os.getenv("HOST_CHROMA")
port = os.getenv("PORT_CHROMA")
provider = os.getenv("CHROMA_CLIENT_AUTH_PROVIDER")
credentials = os.getenv("CHROMA_CLIENT_AUTH_CREDENTIALS")
def view_collections():
    client = chromadb.HttpClient(
        host= host,
        port=443,
        ssl=True,
        headers=None,
        settings=Settings(
            allow_reset=True,
            chroma_client_auth_provider=provider,
            chroma_client_auth_credentials=credentials,
        ),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # This might take a while in the first execution if Chroma wants to download
    # the embedding transformer
    print(client.list_collections())

    st.header("Collections")

    for collection in client.list_collections():
        data = collection.get(include=["embeddings", "documents", "metadatas"])

        ids = data["ids"]
        embeddings = data["embeddings"]
        metadata = data["metadatas"]
        documents = data["documents"]

        df = pd.DataFrame.from_dict(data)
        st.markdown(f"### Collection: **{collection.name}**")
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    try:
        view_collections()
    except Exception as ex:
        print(ex)
        pass
