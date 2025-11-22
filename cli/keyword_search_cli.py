#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    test = subparsers.add_parser("test", help="test functionality")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search(args.query)
            if len(results) == 0:
                print("No results")
            else:
                for index, result in enumerate(results):
                    if index > 4:
                        return
                    print(f"{index+1}. {result["title"]}")
        case "test":
            search("test")
        case _:
            parser.print_help()

def search(search):
    results = []
    with open("./data/movies.json", "r") as file:
        data = json.load(file)
        if "movies" not in data:
            print("ERROR: Key 'movies' not found in dictionary")
            return
        movies = data.get("movies")
        for movie in movies:
            if "title" not in movie:
                print("ERROR: Key 'title' not found in dictionary")
            title = movie.get("title")
            if search in title:
                results.append(movie)
    return sorted(results, key=lambda movie: movie["id"])


if __name__ == "__main__":
    main()
