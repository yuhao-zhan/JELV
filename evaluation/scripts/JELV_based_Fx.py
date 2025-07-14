#!/usr/bin/env python3
import argparse
import os
import sys

# so that we can import the two modules below
sys.path.append(os.path.dirname(__file__))

from JELV_based_F_G import evaluate_cleme
from FluencyScorer.fluency import evaluate_fluency


def main():
    parser = argparse.ArgumentParser(
        description="Compute combined JELV‐based F(x): (1 - γ)·A + γ·B",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s --ref REF.m2 --hyp HYP.m2 [--alpha α] [--gamma γ] [--level system|sentence]"
    )
    parser.add_argument("--ref",   help="Reference M2 file", required=True)
    parser.add_argument("--hyp",   help="Hypothesis M2 file", required=True)
    parser.add_argument("--alpha", help="α for the generalized F (default: 0.5)", type=float, default=0.5)
    parser.add_argument(
        "--gamma", help="γ weight for fluency term (default: 0.5)", type=float, default=0.5
    )
    parser.add_argument(
        "--level", choices=["system", "sentence"], default="system",
        help="Evaluation level:\n"
             "  system   → single combined score\n"
             "  sentence → one combined score per sentence"
    )

    args = parser.parse_args()

    # --- Part A: CLEME‐based metric ---
    cleme_out = evaluate_cleme(
        ref_path=args.ref,
        hyp_path=args.hyp,
        alpha=args.alpha,
        level=args.level
    )

    # --- Part B: Fluency metric ---
    fluency_out = evaluate_fluency(
        hyp_path=args.hyp,
        level=args.level
    )

    # --- Combine ---
    if args.level == "system":
        combined = (1 - args.gamma) * cleme_out + args.gamma * fluency_out
        print(combined)
    else:
        # sentence-level: zip line by line
        for a, b in zip(cleme_out, fluency_out):
            print((1 - args.gamma) * a + args.gamma * b)


if __name__ == "__main__":
    main()
