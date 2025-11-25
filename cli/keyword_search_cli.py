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
            # res = test_search(args.query)
            # searchTerms = tokenizeSearch(args.query) 
            results = search(args.query)
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
            if token in self.index:
                if doc_id not in self.index[token]:
                    self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]

    def test(self, term):
        test = "An elderly English woman Amy Wilkinson (Carole Trangmar-Palmer), almost at her deathbed in London, wants to come down to Madras in search of a young man Ilam Parithi (Arya) whom she last saw on 15 August 1947 to return a thali (traditional wedding threads) of his mother, which he gave her as a sign of stating that she belongs to India and nobody can separate them. However, after a turn of events, she had married another man from her hometown and thus felt that the thali was no longer her property.\nAmy Wilkinson arrives in Madras with her granddaughter Catherine (Lisa Lazarus), equipped only with a picture of Parithi that was taken sixty years ago. Wilkinson interrogates various people about Parithi's whereabouts. In the process, she recalls the events when she had first visited Chennai, and the chain of events that took place:\nA young Amy (Amy Jackson), the daughter of the Madras Presidency Governor, visits Chennai (then called Madharasapattinam) along with her translator Nambi (Cochin Hanifa) and encounters Parithi, whom she calls \"brave man\". Parithi, a member of the dhobi (launderer) clan is also an experienced wrestler who trains under Ayyakanu (Nassar). He openly opposes the British officials who attempt to build a golf course in the dhobi clan's dwelling place. He challenges a cruel racist officer named Robert Ellis (Alexx O'Nell), who is also Amy's suitor, to a wrestling match to decide the fate of his clan's home. Parithi is successful, and Ellis vows revenge.\nFollowing a series of secret meetings between Parithi and Amy, love blossoms between them, and Parithi affectionately calls her \"Duraiyamma\", a polite term of addressing British women. However a major threat comes in the form of independence for India on 15 August 1947, which means that all White officials and their families, including Amy, would have to leave India. On the eve of independence, all of India is celebrating. However Amy and Parithi, determined to be together, run away and are hunted by an angry Ellis and his force. An Indian policeman helps the two of them by hiding them in a clock tower on top of the Madras Central Railway Station, but they are discovered by Ellis. After a fierce fight, Ellis is killed and Parithi is badly wounded. Amy helps Parithi to escape by casting him with a life-raft into the Coovum river, before she is captured and taken back to London. She had never known if Parithi survived, or what his fate was.\nBack in the present, Wilkinson is urgently called back to London to have a life-saving operation. But she is determined to find Parithi and, by chance, encounters a taxi driver who assumes that she would want to visit a charitable trust named Duraiyamma Foundation. The driver shows her around the foundation, which has organisations providing free housing, education and medical care (which were all promised to the dhobi children by the young Amy several years ago). She realizes that the Duraiyamma Foundation was established by Ilam Parithi, and named after her.\nThen When she asks the driver what became of Parithi, he leads her to his tomb, and reveals that he died twelve years ago. She kneels before the tomb and claims the thali (nuptial threads) as her own. She declares \"It's mine!\" before quietly passing away on Parithi's tomb. Her granddaughter mourns for her, and the taxi driver is dumbfounded to learn that the old woman was \"Duraiyamma\" herself. The epilogue shows Parithi and Amy (as they were in their younger days) in the afterlife, depicted as a 1940s-style Madharasapattinam. As the credits roll, a series of montage images are shown, illustrating the transformation of Madharasapattinam of the 1940s to modern-day Chennai."
        test_index = tokenize(test)
        test_index.sort()
        print(test_index)
        return test_index


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

def test_search(search):
    movieIndex = InvertedIndex()
    movieIndex.load()
    res = movieIndex.test(search)
    if "brave" in res:
        print("FOUND IT")


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
            if len(results) >= 5:
                return results
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
