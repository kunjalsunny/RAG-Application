import os
from pydoc import doc
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


 # 1. Loading the files from docs folder
def load_documents(docs_path="docs"):
    
   
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
        print(f"\n Document: {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview:{doc.page_content[:30]}...")
        print(f"Metadata: {doc.metadata}")

    return docs

# 2. Chunking the files
def split_documents(documents, chunk_size=900, chunk_overlap=0):
    print(f"Splitting the documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n ---Chunk{i+1}---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content}")
            print(f"------------------")
    
    return chunks

# 3. Embedding and Storing in VectorDB
def create_vector_store(chunks, persist_directory="db/chromadb"):
    print(f"Create embedding and storing in chromaDB")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("---Create Vector store---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("---Finished create vector store---")
    print(f"Vector store in directory: {persist_directory}")

    return vectorstore

def main():
    #1
    documents = load_documents(docs_path='docs')

    #2
    chunks = split_documents(documents)

    #3
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()