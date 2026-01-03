#!/usr/bin/env python3

import argparse
import math
from lib.keyword_search import InvertedIndex, tokenize

stopwords = []
BM25_K1 = 1.5
BM25_B = 0.75

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build = subparsers.add_parser("build", help="Build inverted index and save to disk")

    tf = subparsers.add_parser("tf", help="Term Frequency")
    tf.add_argument("doc_id", type=int, help="Term frequency search doc_id")
    tf.add_argument("term", type=str, help="Term frequency search term")

    idf = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf.add_argument("term", type=str, help="Inverse Document Frequency search term")

    tfidf = subparsers.add_parser("tfidf", help="TF-IDF")
    tfidf.add_argument("doc_id", type=int, help="doc_id")
    tfidf.add_argument("term", type=str, help="search term")

    bm25idf = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25idf.add_argument("term", type=str, help="Search query")

    bm25tf = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25tf.add_argument("doc_id", type=int, help="Document ID")
    bm25tf.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25tf.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 k1 parameter")
    bm25tf.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="limit")

    test = subparsers.add_parser("test", help="test functionality")
    test.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            ii = InvertedIndex()
            ii.load()
            results = search(args.query, ii)
            for r in results:
                print(f"{r["id"]}  {r["title"]}")
        case "build":
            ii = InvertedIndex()
            print("Building movies index")
            ii.build()
            print("Saving movies index to disk")
            ii.save()
        case "tf":
            ii = InvertedIndex()
            ii.load()
            tf = ii.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            ii = InvertedIndex()
            ii.load()
            doc_count = len(ii.docmap)
            term_doc_count = len(search(args.term, ii))
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            ii = InvertedIndex()
            ii.load()
            doc_count = len(ii.docmap)
            term_doc_count = len(search(args.term, ii))
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            tf = ii.get_tf(args.doc_id, args.term)
            tf_idf = idf * tf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            ii = InvertedIndex()
            ii.load()
            bm25idf = ii.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            ii = InvertedIndex()
            ii.load()
            bm25tf = ii.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            ii = InvertedIndex()
            ii.load()
            bm25_search_results = ii.bm25_search(args.query, args.limit)
            for index, r in enumerate(bm25_search_results.values()):
                print(f"{index}. ({r["movie"]["id"]}) {r["movie"]["title"]} - Score: {r["score"]:.2f}")
        case "test":
            ii = InvertedIndex()
            ii.load()
            docs = ii.get_documents("merida")
            print(docs)
        case _:
            parser.print_help()
            




def test_search(search):
    movieIndex = InvertedIndex()
    movieIndex.load()
    res = movieIndex.test(search)
    if "brave" in res:
        print("FOUND IT")


def search(search, movieIndex):
    # movieIndex = InvertedIndex()
    # movieIndex.load()

    tokens = tokenize(search)
    results = []
    for t in tokens:
        ids = movieIndex.get_documents(t)
        for id in ids:
            results.append(movieIndex.docmap[id])
    if len(results) >= 5:
        return results
    return results

    

if __name__ == "__main__":
    main()
