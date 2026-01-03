import json
import math
import pickle
import os
import string
from typing import Counter
from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75
stopwords = []

class InvertedIndex:
    index = {}
    docmap = {}
    term_frequencies = {}
    doc_lengths = {}
    index_path = "./cache/index.pkl"

    def __init__(self):
        pass

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize(text)
        for token in tokens:
            if token in self.index:
                if doc_id not in self.index[token]:
                    self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]

        self.doc_lengths[doc_id] = len(tokens)

        counter = Counter(tokens)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = counter
        else:
            self.term_frequencies[doc_id].update(counter)

    def __get_avg_doc_length(self):
        doc_lengths = self.doc_lengths.values()
        if len(doc_lengths) == 0:
            return 0.0
        total = sum(doc_lengths)
        # total = map(lambda x, y: x + y, doc_lengths)
        return total / len(doc_lengths) 

    def get_tf(self, doc_id: int, term: str):
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise("ERROR: can only get term frequency of single terms")
        if doc_id not in self.term_frequencies:
            return 0
        tf = self.term_frequencies[doc_id]
        if tokens[0] not in tf:
            return 0
        return tf[tokens[0]]

    def get_documents(self, term: str):
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

        with open(self.index_path, "wb") as f1:
            pickle.dump(self.index, f1)
        with open("./cache/docmap.pkl", "wb") as f2:
            pickle.dump(self.docmap, f2)
        with open("./cache/term_frequencies.pkl", "wb") as f3:
            pickle.dump(self.term_frequencies, f3)
        with open("./cache/doc_lengths.pkl", "wb") as f4:
            pickle.dump(self.doc_lengths, f4)
    
    def load(self):
        if not os.path.exists(self.index_path):
            raise("ERROR: ./cache/index.pkl does not exist")
        if not os.path.exists("./cache/docmap.pkl"):
            raise("ERROR: ./cache/docmap.pkl does not exist")
        if not os.path.exists("./cache/term_frequencies.pkl"):
            raise("ERROR: ./cache/term_frequencies.pkl does not exist")
        if not os.path.exists("./cache/doc_lengths.pkl"):
            raise("ERROR: ./cache/doc_lengths.pkl does not exist")

        with open(self.index_path, "rb")as f1:
            self.index = pickle.load(f1)
        with open("./cache/docmap.pkl", "rb")as f2:
            self.docmap = pickle.load(f2)
        with open("./cache/term_frequencies.pkl", "rb")as f3:
            self.term_frequencies = pickle.load(f3)
        with open("./cache/doc_lengths.pkl", "rb")as f4:
            self.doc_lengths = pickle.load(f4)
    
    def get_bm25_idf(self, term: str):
        # N = total number of docs
        # df = document frequency
        terms = tokenize(term)
        if len(terms) > 1:
            raise("ERROR: Cannot search on more that one term")
        term = terms[0]
        n = len(self.docmap)
        df = 0
        if term in self.index:
            df = len(self.index[term])
        else:
            print("term not found")
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1 = BM25_K1, b = BM25_B):
        # length_norm = 1 - b + b * (doc_length / avg_doc_length)
        # (tf * (k1 + 1)) / (tf + k1)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_bm25(self, doc_id: int, term: str):
        bm25idf = self.get_bm25_idf(term)
        bm25tf = self.get_bm25_tf(doc_id, term)
        return bm25idf * bm25tf

    def bm25_search(self, query: str, limit: int):
        tokens = tokenize(query)
        scores = {}
        for movie in self.docmap.values():
            score = 0
            for token in tokens:
                score += self.get_bm25(movie["id"], token)
            scores[movie["id"]] = score

        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

        retval = {}
        counter = 0
        for key in sorted_scores:
            if counter >= limit:
                break
            retval[key] = {"score": sorted_scores[key], "movie": self.docmap[key]}
            counter += 1
        return retval

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

def get_stopwords():
    global stopwords
    if len(stopwords) > 0:
        return stopwords

    with open("./data/stopwords.txt", "r") as stopwordsFile:
        stopwords = stopwordsFile.read().splitlines()
    return stopwords