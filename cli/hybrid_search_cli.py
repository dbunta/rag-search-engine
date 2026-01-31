import argparse
import json
import time
from lib.hybrid_search import normalize_scores, HybridSearch
import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="")
    weighted_search_parser.add_argument("query", type=str, help="")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="")
    rrf_search_parser.add_argument("query", type=str, help="")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="")
    rrf_search_parser.add_argument( "--enhance", type=str, choices=["spell","rewrite","expand"], help="Query enhancement method")
    rrf_search_parser.add_argument( "--rerank-method", type=str, choices=["individual","batch","cross_encoder"], help="")
    rrf_search_parser.add_argument( "--evaluate", type=bool, help="")

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            documents = []
            with open("./data/movies.json", "r") as file:
                data = json.load(file)
                if "movies" not in data:
                    print("ERROR: Key 'movies' not found in dictionary")
                    return
                documents = data.get("movies")
            hs = HybridSearch(documents)

            query = args.query

            if args.enhance is not None and args.enhance == "spell":
                prompt = f"""Fix any spelling errors in this movie search query.

                        Only correct obvious typos. Don't change correctly spelled words.

                        Query: "{query}"

                        If no errors, return the original query.
                        Corrected:"""
                query = doTheAiStuff(prompt)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            if args.enhance is not None and args.enhance == "rewrite":
                prompt = f"""Rewrite this movie search query to be more specific and searchable.
                    Original: "{query}"

                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep it concise (under 10 words)
                    - It should be a google style search query that's very specific
                    - Don't use boolean logic

                    Examples:

                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                    Rewritten query:"""
                query = doTheAiStuff(prompt)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            if args.enhance is not None and args.enhance == "expand":
                prompt = f"""Expand this movie search query with related terms.

                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    This will be appended to the original query.

                    Examples:

                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"

                    Query: "{query}"
                    """
                query = doTheAiStuff(prompt)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

                
            results = {}

            if args.rerank_method is not None and args.rerank_method == "individual":
                results = hs.rrf_search(query, args.limit*5, args.k)
                new_scores = []
                print("Reranking top 3 results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\r\n")
                for r in results.values():
                    doc = r["document"]
                    prompt = f"""Rate how well this movie matches the search query.

                        Query: "{query}"
                        Movie: {doc.get("title", "")} - {doc.get("document", "")}

                        Consider:
                        - Direct relevance to query
                        - User intent (what taey're looking for)
                        - Content appropriateness

                        Rate 0-10 (10 = perfect match).
                        Give me ONLY the number in your response, no other text or explanation.

                        Score:"""
                    new_result = doTheAiStuff(prompt)
                    new_scores.append({"ranking": int(new_result), "original_result": r})
                    time.sleep(5)
                    # print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                
                sorted_results = sorted(new_scores, key=lambda item: item["ranking"], reverse=True)
                for i, r in enumerate(sorted_results[:args.limit]):
                    print()
                    print(f"{i+1}. {r["original_result"]["document"]["title"]}\r\n   Rerank Score: {r["ranking"]}/10\r\n   RRF Score: {r["original_result"]["score"]:.4f}\r\n   BM25 Rank: {r["original_result"]["bm25_rank"]}, Semantic Rank: {r["original_result"]["semantic_rank"]}\r\n   {r["original_result"]["document"]["description"][:50]}")
                return
            if args.rerank_method is not None and args.rerank_method == "batch":
                results = hs.rrf_search(query, args.limit*5, args.k)
                doc_list_str = ""
                for d in list(results.values()):
                    doc_list_str += f'ID: {d['document']['id']}\r\nTitle: {d['document']["title"]}\r\n\r\n'
                new_scores = []
                print("Reranking top 3 results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\r\n")
                prompt =f"""Rank these movies by relevance to the search query.
                    Query: "{query}"

                    Movies:
                    {doc_list_str}

                    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                    [75, 12, 34, 2, 1]
                    """ 

                new_result = doTheAiStuff(prompt)
                ranks = json.loads(new_result)
                sorted_results = [] 
                print(results.keys())
                for id in list(ranks):
                    sorted_results.append(results[id])

                for i, r in enumerate(sorted_results):
                    print()
                    print(f"{i+1}. {r["document"]["title"]}\r\n   RRF Score: {r["score"]:.4f}\r\n   BM25 Rank: {r["bm25_rank"]:.4f}, Semantic Rank: {r["semantic_rank"]:.4f}\r\n   {r["document"]["description"][:50]}")
                return
            if args.rerank_method is not None and args.rerank_method == "cross_encoder":
                results = hs.rrf_search(query, args.limit*5, args.k)
                pairs = []
                for r in results.values():
                    doc = r['document']
                    pairs.append([query, f"{doc.get('title', '')} - {doc}"])
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)

                # for i,s in enumerate(scores):
                #     print(s)
                #     results["new_score"] = s 
                for i,r in enumerate(results.values()):
                    r["cross_encoder_score"] = scores[i]
                    print(r)
                sorted_results = sorted(results.values(), key=lambda item: item["cross_encoder_score"], reverse=True)

                
                # print(pairs)

                # for i, r in enumerate(scores):
                for i, r in enumerate(sorted_results):
                    print(f"{i+1}. {r["document"]["title"]}\r\n   Cross Encoder Score{r["cross_encoder_score"]:.4f}\r\n   RRF Score: {r["score"]:.4f}\r\n   BM25 Rank: {r["bm25_rank"]:.4f}, Semantic Rank: {r["semantic_rank"]:.4f}\r\n   {r["document"]["description"][:50]}")
                return
        case "weighted-search":
            documents = []
            with open("./data/movies.json", "r") as file:
                data = json.load(file)
                if "movies" not in data:
                    print("ERROR: Key 'movies' not found in dictionary")
                    return
                documents = data.get("movies")
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results.values()):
                print()
                print(f"{i+1}. {r["document"]["title"]}\r\n   Hybrid Score: {r["hybrid"]:.4f}\r\n   BM25: {r["bm25"]:.4f}, Semantic: {r["semantic"]:.4f}\r\n   {r["document"]["description"][:50]}")
        case "normalize":
            if args.scores is not None:
                scores = args.scores
                min_score = min(scores)
                max_score = max(scores)
                if min_score == max_score:
                    normalized_scores = [1.0 for score in scores]
                else:
                    normalized_scores = normalize_scores(scores)
                print("Normalized Scores:", normalized_scores)
        case _:
            parser.print_help()

def doTheAiStuff(prompt:str): 
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    # print(f"Using key {api_key[:6]}...")

    client = genai.Client(api_key=api_key)
    # prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
    # print(response.text)
    # print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    # print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")
    print(response)
    return response.text

if __name__ == "__main__":
    main()


