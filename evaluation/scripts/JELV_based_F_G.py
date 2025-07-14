#!/usr/bin/env python3
import os
import sys

# bring cleme package into PYTHONPATH
sys.path.append(os.path.dirname(__file__) + "/../")

from JELV_based_cleme_cache.cleme import IndependentChunkMetric
from JELV_based_cleme_cache.data import M2DataReader
from JELV_based_cleme_cache.constants import (
    DEFAULT_CONFIG_CORPUS_INDEPENDENT,
    DEFAULT_CONFIG_SENTENCE_INDEPENDENT
)


def evaluate_cleme(ref_path, hyp_path, alpha, level):
    """
    - corpus → returns one float (global F)
    - sentence → returns a list of floats (one F per sent)
    """
    reader = M2DataReader()
    dataset_ref = reader.read(ref_path)
    dataset_hyp = reader.read(hyp_path)
    alpha = float(alpha)

    if level == "corpus":
        metric = IndependentChunkMetric(
            scorer="corpus",
            weigher_config=DEFAULT_CONFIG_CORPUS_INDEPENDENT
        )
        score_dict, _ = metric.evaluate(dataset_hyp, dataset_ref, alpha)
        return score_dict["F"]

    # ─── sentence level ──────────────────────────────────────────────────────
    per_sentence_F = []
    # reuse the SAME corpus config each time
    for ref_sent, hyp_sent in zip(dataset_ref, dataset_hyp):
        metric = IndependentChunkMetric(
            scorer="corpus",
            weigher_config=DEFAULT_CONFIG_CORPUS_INDEPENDENT
        )
        # note: we wrap each single-sentence as length-1 lists
        score_dict, _ = metric.evaluate([hyp_sent], [ref_sent], alpha)
        per_sentence_F.append(score_dict["F"])

    return per_sentence_F