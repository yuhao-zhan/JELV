#!/usr/bin/env python3
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
import logging
import os
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_VERBOSITY"] = "error"

from FluencyScorer.utils import parse_m2_blocks, apply_edits_to_sentence

warnings.filterwarnings("ignore")
# Load once
MODEL_NAME = "gpt2"
_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
_tokenizer = GPT2Tokenizer.from_pretrained(
    MODEL_NAME,
    clean_up_tokenization_spaces=False
)


def calculate_cross_entropy(sentence: str) -> float:
    """
    Compute average cross-entropy loss H(x) for the sentence under GPT-2.
    """
    tokens = _tokenizer.encode(sentence, return_tensors="pt")
    if tokens.numel() == 0:
        # either return a default high/low score or skip
        return float("nan")
    with torch.no_grad():
        outputs = _model(tokens, labels=tokens)
    return outputs.loss.item()


def fluency_score_sentence(sentence: str) -> float:
    """
    f(x) = 4 / (1 + H(x))
    """
    Hx = calculate_cross_entropy(sentence)
    return 4.0 / (1.0 + Hx)


def evaluate_fluency(hyp_path: str, level: str):
    """
    Convert hypothesis M2 → list of corrected sentences (using annotator_id=0),
    compute f(x) per sentence, then return either a single average (corpus)
    or the list (sentence).
    """
    blocks = parse_m2_blocks(hyp_path)
    sentences = []

    for blk in blocks:
        toks = blk["sentence_tokens"]
        # group by annotator_id
        by_id = {}
        for ann in blk["annotations"]:
            aid, token_range, edit_type, corr = ann
            by_id.setdefault(aid, []).append((token_range, edit_type, corr))

        # apply only annotator 0 edits (fallback to source if none)
        if 0 in by_id:
            sent = apply_edits_to_sentence(toks, by_id[0])
        else:
            sent = " ".join(toks)
        sentences.append(sent)

    scores = [fluency_score_sentence(s) for s in sentences]

    if level == "corpus":
        return sum(scores) / len(scores) if scores else 0.0
    else:
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute fluency f(x) per sentence or corpus"
    )
    parser.add_argument(
        "--hyp", required=True,
        help="Hypothesis M2 file to convert and score"
    )
    parser.add_argument(
        "--level", default="corpus", choices=["corpus", "sentence"],
        help="Output a single average (corpus) or one f(x) per line (sentence)"
    )
    args = parser.parse_args()

    out = evaluate_fluency(args.hyp, args.level)
    if args.level == "corpus":
        print(out)
    else:
        for val in out:
            print(val)
