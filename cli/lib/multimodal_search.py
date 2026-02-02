import json
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

def search_with_image(path:str):
    documents = []
    with open("./data/movies.json", "r") as file:
        data = json.load(file)
        if "movies" not in data:
            print("ERROR: Key 'movies' not found in dictionary")
            return
        documents = data.get("movies")
    ms = MultimodalSearch(documents=documents)
    scores = ms.search_with_image(path)
    return scores

def verify_image_embedding(path:str):
    ms = MultimodalSearch()
    embedding = ms.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class MultimodalSearch:
    def __init__(self, documents:list, model_name:str="clip-ViT-B-32"):
        self.texts = []
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path:str):
        img = Image.open(path)
        embedding = self.model.encode(img)
        return embedding
    
    def search_with_image(self, path:str):
        scores = []
        image_embedding = self.embed_image(path)
        for i,embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(image_embedding, embedding)
            scores.append({"score": score, "doc": self.documents[i]})
        sorted_scores = sorted(scores, key=lambda item: item['score'], reverse=True)
        return sorted_scores[:5]

