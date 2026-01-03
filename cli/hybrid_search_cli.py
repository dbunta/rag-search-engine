import argparse
import json
from lib.hybrid_search import normalize_scores, HybridSearch


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
            results = hs.rrf_search(args.query, args.limit, args.k)
            for i, r in enumerate(results.values()):
                print()
                print(f"{i+1}. {r["document"]["title"]}\r\n   RRF Score: {r["score"]:.4f}\r\n   BM25 Rank: {r["bm25_rank"]:.4f}, Semantic Rank: {r["semantic_rank"]:.4f}\r\n   {r["document"]["description"][:50]}")
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


if __name__ == "__main__":
    main()