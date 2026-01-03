import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunked_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def rrf_search(self, query, limit, k):
        bm25_results = self._bm25_search(query, limit)
        semantic_results = self.semantic_search.search_chunks(query, limit)
        scores = {}
        counter = 0
        for doc_id, keyword_result in bm25_results.items():
            counter += 1
            if doc_id in scores:
                scores[doc_id]["bm25_rank"] = counter
            else:
                scores[doc_id] = {"bm25_rank": counter, "semantic_rank": 0, "score": 0.0}

        counter = 0
        for semantic_result in semantic_results:
            counter += 1 
            if semantic_result["id"] in scores:
                scores[semantic_result["id"]]["semantic_rank"] = counter
            else:
                scores[semantic_result["id"]] = {"semantic_rank": counter, "bm25_rank": 0, "score": 0.0}

        for key, value in scores.items():
            score = 0
            if scores[key]["semantic_rank"] > 0:
                score += 1 / (k + value["semantic_rank"])
            if scores[key]["bm25_rank"] > 0:
                score += 1 / (k + value["bm25_rank"])

            scores[key]["score"] = score
            scores[key]["document"] = self.semantic_search.document_map[key]
        
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True))
        return sorted_scores

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha=0.5, limit=5):
        # limit = limit * 500
        bm25_results = self._bm25_search(query, limit)
        semantic_results = self.semantic_search.search_chunks(query, limit)
        normalized_bm25_scores = normalize_scores({id: result["score"] for id, result in bm25_results.items()})
        normalized_semantic_scores = normalize_scores({result["id"]: result["score"] for result in semantic_results})

        scores = {}
        for doc_id, keyword_score in normalized_bm25_scores.items():
            if doc_id in scores:
                scores[doc_id]["bm25"] = keyword_score
            else:
                scores[doc_id] = {"bm25": keyword_score, "semantic": 0.0, "hybrid": 0.0}

        for doc_id, semantic_score in normalized_semantic_scores.items():
            if doc_id in scores:
                scores[doc_id]["semantic"] = semantic_score
            else:
                scores[doc_id] = {"bm25": 0.0, "semantic": semantic_score, "hybrid": 0.0}
        
        for key, value in scores.items():
            scores[key]["hybrid"] = self.hybrid_score(value["bm25"], value["semantic"], alpha)
            scores[key]["document"] = self.semantic_search.document_map[key]

        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]["hybrid"], reverse=True))
        return sorted_scores
            



    # def rrf_search(self, query, k, limit=10):
    #     raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_scores(doc_scores):
    min_score = min(doc_scores.values())
    max_score = max(doc_scores.values())
    if min_score == max_score:
        return {id: 1.0 for id, _ in doc_scores.items()}
    return {
        id: (score - min_score) / (max_score - min_score) if max_score > min_score else 0.0
        for id, score in doc_scores.items()
    }