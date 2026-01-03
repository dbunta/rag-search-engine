import json
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.model: SentenceTransformer
    
    def generate_embeddings(self, text: str):
        text = text.strip()
        if len(text) == 0:
            raise ValueError("Cannot generate embedding for an empty string")
        embeddings = self.model.encode(text)
        return embeddings
    
    def build_embeddings(self, documents):
        self.documents = documents
        documents_str = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            documents_str.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(documents_str, show_progress_bar=True)
        with open("./cache/movie_embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        documents_str = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            documents_str.append(f"{doc['title']}: {doc['description']}")
        if os.path.exists("cache/movie_embeddings.npy"):
            with open("cache/movie_embeddings.npy", "rb") as f:
                self.embeddings = np.load(f)
                if len(self.embeddings) == len(self.documents):
                    return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        scores = []
        query_embedding = self.generate_embeddings(query)
        for index, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            scores.append((similarity_score, self.documents[index]))

        sorted_scores = sorted(scores, key=lambda item:item[0], reverse=True)
        return sorted_scores[:limit]

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        
        chunk_metadata = []
        all_chunks = []

        for i, document in enumerate(documents):
            self.document_map[document["id"]] = document

            if not document["description"]:
                continue

            chunks = semantic_chunk(document["description"], 4, 1)
            for j, chunk in enumerate(chunks):
                print(f"Processing: Document {i+1}/{len(documents)}", end="\r")
                all_chunks.append(chunk)
                chunk_metadata.append({"movie_idx": i, "chunk_idx": j, "total_chunks": len(chunks)})
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        with open("./cache/chunk_embeddings.npy", "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open("./cache/chunk_metadata.json", "w") as f2:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f2, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunked_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open("cache/chunk_metadata.json", "r") as f2:
                self.chunk_metadata = json.load(f2)
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        chunk_scores = [] 
        movie_scores = {}
        query = query.strip()
        if len(query) == 0:
            return []
        query_embedding = self.generate_embeddings(query)

        for index, doc_embedding in enumerate(self.chunk_embeddings):
            print(f"Searching: Document {index+1}/{len(self.chunk_embeddings)}", end="\r")
            score = cosine_similarity(query_embedding, doc_embedding)
            chunk_metadata_2 = self.chunk_metadata["chunks"][index]
            movie_idx = chunk_metadata_2["movie_idx"]
            chunk_idx = chunk_metadata_2["chunk_idx"]
            chunk_scores.append({"chunk_idx":chunk_idx, "movie_idx":movie_idx, "score":score})
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        sorted_movie_scores_2 = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
        sorted_movie_scores = dict(sorted_movie_scores_2[:limit])
        results = []
        for index, movie_score_key in enumerate(sorted_movie_scores.keys()):
            document = self.documents[movie_score_key]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "document": document["description"][:100],
                    "score": round(sorted_movie_scores[movie_score_key], 4),
                    "metadata": chunk_metadata_2 or {}
                }
            )
        return results

def verify_embeddings():
    ss = SemanticSearch()
    documents = []
    with open("./data/movies.json", "r") as file:
        data = json.load(file)
        if "movies" not in data:
            print("ERROR: Key 'movies' not found in dictionary")
            return
        documents = data.get("movies")
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embeddings(text)
    print(embedding)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1
def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    chunks = []
    print(text)
    text = text.strip()
    if len(text) == 0:
        return chunks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    i = 0
    n_sentences = len(sentences)

    if n_sentences == 1 and not sentences[0].endswith((".", "!", "?")):
        return [text]

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]

        for cs in chunk_sentences:
            cs = cs.strip()

        if chunks and len(chunk_sentences) <= overlap:
            break
        if len(chunk_sentences) == 0:
            continue

        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap

    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    if len(chunks) == 0:
        return []
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")