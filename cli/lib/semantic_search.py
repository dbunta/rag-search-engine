import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    model: SentenceTransformer
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        # print(f"Model loaded: {model}")
        # print(f"Max sequence length: {model.max_seq_length}")
        # model.encode(text)
    
    def generate_embeddings(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("Cannot generate embedding for an empty string")
        embeddings = self.model.encode(text)
        # return embeddings[0]
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

        # sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        # sorted_scores = scores
        # scores = scores.sort(reverse=True)
        sorted_scores = sorted(scores, key=lambda item:item[0], reverse=True)
        return sorted_scores[:limit]

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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)