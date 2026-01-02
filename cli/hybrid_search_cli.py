import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores is not None:
                scores = args.scores
                min_score = min(scores)
                max_score = max(scores)
                if min_score == max_score:
                    normalized_scores = [1.0 for score in scores]
                else:
                    normalized_scores = [
                        (score - min_score) / (max_score - min_score) if max_score > min_score else 0.0
                        for score in scores
                    ]
                print("Normalized Scores:", normalized_scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()