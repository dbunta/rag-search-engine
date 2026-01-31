import argparse
import json
from lib.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    test_cases = []
    with open("./data/golden_dataset.json", "r") as file:
        data = json.load(file)
        test_cases = data["test_cases"]
        file.close()
    with open("./data/movies.json", "r") as file:
        data = json.load(file)
        if "movies" not in data:
            print("ERROR: Key 'movies' not found in dictionary")
            return
        documents = data.get("movies")
    hs = HybridSearch(documents)

    for tc in test_cases:
    # tc = test_cases[4]
        res = hs.rrf_search(tc["query"], limit, 60)
        values = res.values()
        doc_list = [r["document"] for r in values]
        title_list = [d["title"].lower() for d in doc_list]
        # print(title_list)
        rel_docs = []
        # print(title_list)
        # print(tc["relevant_docs"])

        for rel in tc["relevant_docs"]:
            if rel.lower() in title_list:
                rel_docs.append(rel)
        # precision = relevant_retrieved / total_retrieved
        precision = len(rel_docs) / len(title_list)
        # recall = relevant_retrieved / total_relevant
        recall = len(rel_docs) / len(tc["relevant_docs"])
        # f1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall)
        tc["precision"] = precision
        tc["relevant"] = rel_docs
        tc["retrieved"] = title_list
        tc["recall"] = recall
        tc["f1"] = f1
    # print()
    # print(rel_docs)
    # print(f"k=6")
    print()
    for tc in test_cases:
        print(f"- Query: {tc["query"]}\r\n  - Precision@{limit}: {tc["precision"]:.4f}\r\n  - Recall@{limit}: {tc["recall"]:.4f}\r\n  - F1 Score: {tc["f1"]:.4f}\r\n  - Retrieved: {", ".join(tc["retrieved"])}\r\n  - Relevant: {", ".join(tc["relevant"])}\r\n\r\n")
        



if __name__ == "__main__":
    main()