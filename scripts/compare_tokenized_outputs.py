import argparse
from pathlib import Path
import sys


def _read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip() for line in handle]


def main():
    parser = argparse.ArgumentParser(
        description="Compare two tokenized output files line-by-line."
    )
    parser.add_argument("reference_file", type=Path)
    parser.add_argument("candidate_file", type=Path)
    args = parser.parse_args()

    reference_lines = _read_lines(args.reference_file)
    candidate_lines = _read_lines(args.candidate_file)

    if len(reference_lines) != len(candidate_lines):
        print(
            f"Line count differs: reference={len(reference_lines)}, "
            f"candidate={len(candidate_lines)}"
        )
        sys.exit(1)

    for index, (reference_line, candidate_line) in enumerate(
        zip(reference_lines, candidate_lines),
        start=1,
    ):
        if reference_line != candidate_line:
            print(f"Mismatch at line {index}")
            print(f"Reference: {reference_line[:200]}")
            print(f"Candidate: {candidate_line[:200]}")
            sys.exit(1)

    print(
        f"Files match exactly: {args.reference_file} == {args.candidate_file} "
        f"({len(reference_lines)} lines)"
    )


if __name__ == "__main__":
    main()
