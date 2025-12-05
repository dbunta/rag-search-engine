#!/usr/bin/env python3

import argparse
import json

import lib.semantic_search as ss

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify model")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", type=str, help="text to embed")

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query")
    embed_query_parser.add_argument("query", type=str, help="query to embed")

    search_parser = subparsers.add_parser("search", help="search")
    search_parser.add_argument("query", type=str, help="query to search")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="number of results")

    chunk_parser = subparsers.add_parser("chunk", help="chunk")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="size of chunks")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=2, help="size of chunks")

    args = parser.parse_args()
    match args.command:
        case "verify":
            ss.verify_model()
        case "verify_embeddings":
            ss.verify_embeddings()
        case "embed_text":
            ss.embed_text(args.text)
        case "embedquery":
            ss.embed_query_text(args.query)
        case "search":
            ss2 = ss.SemanticSearch()
            documents = []
            with open("./data/movies.json", "r") as file:
                data = json.load(file)
                if "movies" not in data:
                    print("ERROR: Key 'movies' not found in dictionary")
                    return
                documents = data.get("movies")
            embeddings = ss2.load_or_create_embeddings(documents)
            results = ss2.search(args.query, args.limit)
            for i, r in enumerate(results):
                print(f"{i}. {r[1]["title"]} (score: {r[0]:.2f})\r\n{r[1]["description"][:20]}...")
        case "chunk":
            chunks = list(chunk_text(args.text.rsplit(), args.chunk_size, args.overlap))

            # for i, word in enumerate(words):
            #     chunk += f"{word} "
            #     if counter == chunk_size - 1 or len(words) - 1 == i:
            #         chunks.append(chunk)
            #         chunk = ""
            #         counter = 0
            #     else:
            #         counter += 1
            
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                c = " ".join(chunk)
                print(f"{i+1}. {c}")
            pass
        case _:
            parser.print_help()

def chunk_text(text, chunk_size, overlap):
    for i in range(0, len(text), chunk_size):
        if i == 0:
            yield text[i:i + chunk_size]
        else:
            yield text[i-overlap:i + chunk_size]



if __name__ == "__main__":
    main()