#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    test = subparsers.add_parser("test", help="test functionality")
    test.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # searchTerms = tokenizeSearch(args.query) 
            results = search(args.query)
            # results = search(args.query)
            if len(results) == 0:
                print("No results")
            else:
                for index, result in enumerate(results):
                    if index > 4:
                        return
                    print(f"{index+1}. {result["title"]}")
        case "test":
            terms = tokenize(args.query)
            test = ["its", "magic"]
            test2 = any(item in terms for item in test)
            print(test2)
        case _:
            parser.print_help()

def tokenize(search, stopwords):
    search = search.lower()
    dict = {}
    for punc in string.punctuation:
        dict[punc] = None
    table = str.maketrans(dict)
    search = search.translate(table)
    searchTerms = search.split()
    searchTerms = [t for t in searchTerms if t not in stopwords]
    stemmer = PorterStemmer()
    for index, st in enumerate(searchTerms):
        searchTerms[index] = stemmer.stem(st)

    return searchTerms

def search(search):
    results = []
    stopwords = []
    movies = {}

    with open("./data/stopwords.txt", "r") as stopwordsFile:
        stopwords = stopwordsFile.read().splitlines()

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
        searchTokens = tokenize(search, stopwords)
        targetTokens = tokenize(title, stopwords)
        previousSearchToken = ""
        isAdded = False
        for searchToken in searchTokens:
            if searchToken != previousSearchToken and isAdded:
                isAdded = False
                previousSearchToken = searchToken
                continue
            for targetToken in targetTokens:
                if searchToken in targetToken:
                    results.append(movie)
                    isAdded = True
                    break
            previousSearchToken = searchToken
    return sorted(results, key=lambda movie: movie["id"])


if __name__ == "__main__":
    main()
