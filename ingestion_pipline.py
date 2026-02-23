import os
from pydoc import doc
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    
    # 1. Loading the files from docs folder
    print(f"Loading the documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The Directory {docs_path} does not exist.")

    loader = PyPDFDirectoryLoader(
        path=docs_path,
        glob="*.pdf",
    )    
    docs = loader.load()
    print(f"Loaded {len(docs)} document pages")

    for i, doc in enumerate(docs[:2]):
        print(f"\n Document: {i}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview:{doc.page_content[:30]}...")
        print(f"Metadata: {doc.metadata}")

    return docs

    # 2. Chunking the files
    def split_documents(documents, chunk_size=1000, chunk_overlap=0):
        pass


    # 3. Embedding and Storing in VectorDB

if __name__ == "__main__":
    documents = load_documents(docs_path='docs')