import os
import hashlib
import math
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()



PROJECT_DIR = Path(__file__).resolve().parent

class LocalHashEmbeddings(Embeddings):
    """Create deterministic, dependency-free lexical embeddings."""

    dimensions = 1536

    def _embed(self, text):
        tokens = re.findall(r"\w+", text.lower())
        features = [f"word:{token}" for token in tokens]
        features.extend(
            f"bigram:{left}_{right}" for left, right in zip(tokens, tokens[1:])
        )

        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            vector[index] += 1.0 if value & (1 << 8) else -1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def load_documents(docs_path="document"):
    """Load non-empty text files from a directory."""
    path = Path(docs_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.is_dir():
        raise FileNotFoundError(
            f"The document directory '{path}' does not exist. "
            "Create it and add your .txt files."
        )

    print(f"Loading documents from {path}...")
    documents = [
        Document(
            page_content=file_path.read_text(encoding="utf-8"),
            metadata={"source": str(file_path)},
        )
        for file_path in sorted(path.rglob("*.txt"))
    ]
    documents = [
        document
        for document in documents
        if document.page_content and document.page_content.strip()
    ]

    if not documents:
        raise FileNotFoundError(f"No non-empty .txt files found in '{path}'.")

    for index, document in enumerate(documents[:2], start=1):
        print(f"\nDocument {index}:")
        print(f"Source: {document.metadata['source']}")
        print(f"Content length: {len(document.page_content)} characters")
        print(f"Metadata: {document.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with optional overlap."""
    if not documents:
        raise ValueError("Cannot split documents: no documents were loaded.")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller than it.")

    print("Splitting documents into chunks...")
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(chunk.page_content)
        print("_" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
    print(f"Created {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Embed chunks and persist them in ChromaDB."""
    if not chunks:
        raise ValueError("Cannot create a vector store: no document chunks were created.")

    persist_path = PROJECT_DIR / persist_directory
    print("Creating embeddings and storing them in ChromaDB...")
    if os.getenv("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("Using OpenAI embeddings.")
    else:
        embeddings = LocalHashEmbeddings()
        print(
            "OPENAI_API_KEY is not set; using local lexical embeddings. "
            "Set the key in .env for semantic OpenAI embeddings."
        )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store created and saved to {persist_path}.")
    return vectorstore


def main():
    print("Main Function")
    documents = load_documents()
    chunks = split_documents(documents)
    create_vector_store(chunks)
    return chunks


if __name__ == "__main__":
    main()

