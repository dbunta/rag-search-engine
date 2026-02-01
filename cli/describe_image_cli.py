
import argparse
import mimetypes
import types
from google.genai import types

from lib.utils import doTheAiStuff2



def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    # subparsers = parser.add_subparsers(dest="command", help="Available commands")


    parser.add_argument("--image", type=str, help="path to image")
    parser.add_argument("--query", type=str, help="query string")
    # rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # summarize_parser = subparsers.add_parser("summarize", help="")
    # summarize_parser.add_argument("query", type=str, help="")
    # summarize_parser.add_argument("--limit", type=int, default=5, help="")

    # citation_parser = subparsers.add_parser("citations", help="")
    # citation_parser.add_argument("query", type=str, help="")
    # citation_parser.add_argument("--limit", type=int, default=5, help="")

    # question_parser = subparsers.add_parser("question", help="")
    # question_parser.add_argument("question", type=str, help="")
    # question_parser.add_argument("--limit", type=int, default=5, help="")
    args = parser.parse_args()
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    if not mime:
        print("bad")
        exit()
    img = []
    with open(args.image, "rb") as file:
        img = file.read()
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary"""
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]
    response = doTheAiStuff2(parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")



if __name__ == "__main__":
    main()