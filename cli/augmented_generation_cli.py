import argparse
import json

from lib.utils import doTheAiStuff
from lib.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="")
    summarize_parser.add_argument("query", type=str, help="")
    summarize_parser.add_argument("--limit", type=int, default=5, help="")

    citation_parser = subparsers.add_parser("citations", help="")
    citation_parser.add_argument("query", type=str, help="")
    citation_parser.add_argument("--limit", type=int, default=5, help="")

    question_parser = subparsers.add_parser("question", help="")
    question_parser.add_argument("question", type=str, help="")
    question_parser.add_argument("--limit", type=int, default=5, help="")
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = []
            print("Loading movies.json")
            with open("./data/movies.json", "r") as file:
                data = json.load(file)
                if "movies" not in data:
                    print("ERROR: Key 'movies' not found in dictionary")
                    return
                documents = data.get("movies")
                print("movies.json loaded")
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, 5, 60)
            docs = []
            for r in results.values():
                docs.append(r["document"])
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {docs}

                Provide a comprehensive answer that addresses the query:"""
            ragResponse = doTheAiStuff(prompt)
            # print(aiResults)
            print("Search Results:")
            for d in docs:
                print(f"\t- {d["title"]}")
            print("RAG Response:")
            print(ragResponse)
        case "summarize":
            query = args.query
            limit = args.limit
            documents = []
            print("Loading movies.json")
            with open("./data/movies.json", "r") as file:
                data = json.load(file)
                if "movies" not in data:
                    print("ERROR: Key 'movies' not found in dictionary")
                    return
                documents = data.get("movies")
                print("movies.json loaded")
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, limit, 60)
            prompt = f"""
                Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results:
                {results}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
                """
            ragResponse = doTheAiStuff(prompt)
            print("Search Results:")
            for r in results.values():
                print(f'\t- {r["document"]["title"]}')

            print("")
            print("LLM Response:")
            print(ragResponse)
        case "citations":
            query = args.query
            limit = args.limit
            documents = []
            print("Loading movies.json")
            with open("./data/movies.json", "r") as file:
                data = json.load(file)
                if "movies" not in data:
                    print("ERROR: Key 'movies' not found in dictionary")
                    return
                documents = data.get("movies")
                print("movies.json loaded")
            hs = HybridSearch(documents)
            results = hs.rrf_search(query, limit, 60)
            documents = []
            for r in results.values():
                documents.append(r["document"])
            prompt = f"""Answer the question or provide information based on the provided documents.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                Query: {query}

                Documents:
                {documents}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources using [1], [2], etc. format when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""
            ragResponse = doTheAiStuff(prompt) 
            print("Search Results:")
            for r in results.values():
                print(f'\t- {r["document"]["title"]}')

            print("")
            print("LLM Response:")
            print(ragResponse)
        case "question":
            question = args.question
            limit = args.limit
            documents = []
            print("Loading movies.json")
            with open("./data/movies.json", "r") as file:
                data = json.load(file)
                if "movies" not in data:
                    print("ERROR: Key 'movies' not found in dictionary")
                    return
                documents = data.get("movies")
                print("movies.json loaded")
            hs = HybridSearch(documents)
            results = hs.rrf_search(question, limit, 60)
            documents = []
            for r in results.values():
                documents.append(r["document"])
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Question: {question}

            Documents:
            {results}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
            ragResponse = doTheAiStuff(prompt) 
            print("Search Results:")
            for r in results.values():
                print(f'\t- {r["document"]["title"]}')

            print("")
            print("Answer:")
            print(ragResponse)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()