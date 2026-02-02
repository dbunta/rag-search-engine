import argparse

from lib.multimodal_search import search_with_image, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verifyParser = subparsers.add_parser("verify_image_embedding")
    verifyParser.add_argument("path", type=str, help="")

    imageSearchParser = subparsers.add_parser("image_search")
    imageSearchParser.add_argument("path", type=str, help="")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            results = search_with_image(args.path)
            for i,r in enumerate(results):
                print(f"{i+1}: {r['doc']['title']} (similarity: {r['score']:.3f})\r\n\t{r['doc']['description']}")



if __name__ == "__main__":
    main()