from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory="db/chromadb"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)
print("Vector count:", db._collection.count())

model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(user_question):
    print(f"\n---You asked: {user_question}----")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question ")
        ] + chat_history + [
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k":5})
    docs = retriever.invoke(search_question)

    if not docs:
        answer = "I don't know based on the documents."
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))
        print(f"Answer: {answer}")
        return answer


    print(f"Found: {len(docs)} relevant documents")
    for i, doc in enumerate(docs,1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Doc {i}: {preview}...")

    combined_input = f"""Based on the following documents, please answer this questions: {user_question}

    Documents:
    {chr(10).join([f"-{doc.page_content}" for doc in docs])}

    If the answer is not in the context, say you don't know.
    """

    messages = [
        SystemMessage(content="Answer ONLY using the provided Documents. If not found, say you don't know.")
        ] + chat_history + [
            HumanMessage(content=combined_input)  
        ]


    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")

    return answer


def start_chat():
    print("Ask me question: Type 'quit' to exit")

    while True:
        question = input("\n Your question: ")

        if question.lower() == "quit":
            print("Goodbye")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()