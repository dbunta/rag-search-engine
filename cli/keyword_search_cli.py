#!/usr/bin/env python3

import argparse
import json
import pickle
import os
import string
from nltk.stem import PorterStemmer

stopwords = []

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build = subparsers.add_parser("build", help="Build inverted index and save to disk")
    # build.add_argument("query", type=str, help="Search query")

    test = subparsers.add_parser("test", help="test functionality")
    test.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # searchTerms = tokenizeSearch(args.query) 
            results = search(args.query)
            # results = search(args.query)
            for r in results:
                print(f"{r["id"]}  {r["title"]}")
            # if len(results) == 0:
            #     print("No results")
            # else:
            #     for index, result in enumerate(results):
            #         if index > 4:
            #             return
            #         # print(f"{index+1}. {result["title"]}")
            #         print(f"{result["id"]} {result["title"]}")
        case "build":
            ii = InvertedIndex()
            print("Building movies index")
            ii.build()
            print("Saving movies index to disk")
            ii.save()
        case "test":
            # terms = tokenize(args.query)
            ii = InvertedIndex()
            ii.load()
            docs = ii.get_documents("merida")
            print(docs)
        case _:
            parser.print_help()
            
class InvertedIndex:
    index = {}
    docmap = {}

    def __init__(self):
        pass

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token in self.index and doc_id not in self.index[token]:
                self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]

    def get_documents(self, term):
        term = term.lower()
        if term in self.index:
            ids = self.index[term]
            ids.sort()
            return ids
        return []

    def build(self):
        with open("./data/movies.json", "r") as file:
            data = json.load(file)
            if "movies" not in data:
                print("ERROR: Key 'movies' not found in dictionary")
                return
            movies = data.get("movies")

        total = len(movies)
        for index, movie in enumerate(movies):
            if "title" not in movie:
                print("ERROR: Key 'title' not found in dictionary")
            if "description" not in movie:
                print("ERROR: Key 'description' not found in dictionary")
            if "id" not in movie:
                print("ERROR: Key 'id' not found in dictionary")
            title = movie.get("title")
            description = movie.get("description")
            id = movie.get("id")

            print(f"{index+1}/{total}", end='\r')
            self.__add_document(id, f"{title} {description}")
            self.docmap[id] = movie
    
    def save(self):
        if not os.path.isdir("./cache"):
            os.mkdir("./cache")

        with open("./cache/index.pkl", "wb") as f1:
            pickle.dump(self.index, f1)
        with open("./cache/docmap.pkl", "wb") as f2:
            pickle.dump(self.docmap, f2)
    
    def load(self):
        if not os.path.exists("./cache/index.pkl"):
            raise("ERROR: ./cache/index.pkl does not exist")
        if not os.path.exists("./cache/docmap.pkl"):
            raise("ERROR: ./cache/docmap.pkl does not exist")

        with open("./cache/index.pkl", "rb")as f1:
            self.index = pickle.load(f1)
        with open("./cache/docmap.pkl", "rb")as f2:
            self.docmap = pickle.load(f2)



def tokenize(search):
    search = search.lower()
    dict = {}
    for punc in string.punctuation:
        dict[punc] = None
    table = str.maketrans(dict)
    search = search.translate(table)
    searchTerms = search.split()
    searchTerms = [t for t in searchTerms if t not in get_stopwords()]
    stemmer = PorterStemmer()
    for index, st in enumerate(searchTerms):
        searchTerms[index] = stemmer.stem(st)

    return searchTerms

def search(search):
    # results = []
    # movies = {}

    movieIndex = InvertedIndex()
    movieIndex.load()

    tokens = tokenize(search)
    results = []
    for t in tokens:
        ids = movieIndex.get_documents(t)
        for id in ids:
            results.append(movieIndex.docmap[id])
            # if len(results) >= 5:
            #     return results
    return results


    # with open("./data/movies.json", "r") as file:
    #     data = json.load(file)
    #     if "movies" not in data:
    #         print("ERROR: Key 'movies' not found in dictionary")
    #         return
    #     movies = data.get("movies")

    # for movie in movies:
    #     if "title" not in movie:
    #         print("ERROR: Key 'title' not found in dictionary")
    #     title = movie.get("title")
    #     searchTokens = tokenize(search)
    #     targetTokens = tokenize(title)
    #     previousSearchToken = ""
    #     isAdded = False
    #     for searchToken in searchTokens:
    #         if searchToken != previousSearchToken and isAdded:
    #             isAdded = False
    #             previousSearchToken = searchToken
    #             continue
    #         for targetToken in targetTokens:
    #             if searchToken in targetToken:
    #                 results.append(movie)
    #                 isAdded = True
    #                 break
    #         previousSearchToken = searchToken
    # return sorted(results, key=lambda movie: movie["id"])

def get_stopwords():
    global stopwords
    if len(stopwords) > 0:
        return stopwords

    with open("./data/stopwords.txt", "r") as stopwordsFile:
        stopwords = stopwordsFile.read().splitlines()
    return stopwords
    

if __name__ == "__main__":
    main()
