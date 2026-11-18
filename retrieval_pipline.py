import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ingestion_pipline import LocalHashEmbeddings


load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = PROJECT_DIR / "db" / "chroma_db"


def get_embedding_model():
    """Return the same embedding backend used during ingestion."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return LocalHashEmbeddings()


def retrieve_documents(query: str, k: int = 3):
    """Load the persisted Chroma vector store and retrieve relevant chunks."""
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{PERSIST_DIRECTORY}'. "
            "Run ingestion_pipline.py first to create it."
        )

    vector_store = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    documents = vector_store.similarity_search(query, k=k)

    unique_documents = []
    seen_content = set()
    for document in documents:
        if document.page_content not in seen_content:
            unique_documents.append(document)
            seen_content.add(document.page_content)
    return unique_documents


def main():
    query = "What does Acme Analytics help teams understand?"
    relevant_docs = retrieve_documents(query)

    print(f"User Query: {query}")
    print("___Context___")
    if not relevant_docs:
        print("No relevant documents found.")
        return

    context = "\n\n".join(document.page_content for document in relevant_docs)
    for index, document in enumerate(relevant_docs, start=1):
        print(f"Document {index}:\n{document.page_content}\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to generate an answer with OpenAI.")
        return

    prompt = (
        f"Answer the question using only the information in these documents.\n"
        f"If the documents do not contain the answer, say so.\n\n"
        f"Question: {query}\n\nDocuments:\n{context}"
    )
    answer = ChatOpenAI(model="gpt-4o").invoke(prompt)
    print("---Generated response---")
    print(answer.content)


if __name__ == "__main__":
    main()
