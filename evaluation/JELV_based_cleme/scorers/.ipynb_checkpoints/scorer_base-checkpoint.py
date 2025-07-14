"""
`Scorer` is an abstract class that enforces the implementation of a set
of abstract methods. This way, a correctly implemented metric will work
seamlessly with the rest of the codebase.

Scorer                                        # Abstract Scorer Class
  ├── SystemScorer                            # Corpus-level Scorer
  └── SentenceScorer                          # Sentence-level Scorer
  └── SentenceScorerForGLEU                   # Sentence-level Scorer for GLEU.txt
  └── SentenceScorerForGLEU                   # Sentence-level Scorer for GLEU.txt
"""

import abc
from dataclasses import dataclass, field

from typing import Any, List, Dict


def compute_f(tp, fp, fn, beta=0.5):
    p = float(tp) / (tp + fp) if fp else 1.0
    r = float(tp) / (tp + fn) if fn else 1.0
    f = float((1 + (beta ** 2)) * p * r) / (((beta ** 2) * p) + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

def compute_generalized_f(tp, fp, fn, alpha, OC, beta=0.5):
    # print(f"tp: {tp}\t fp: {fp}\t fn: {fn}\t OC: {OC}")
    # alpha = 0.4
    
    p = float(tp) / (tp + fp) if fp else 1.0
    r = float(tp) / (tp + fn) if fn else 1.0
    f_modified = float((1 + (beta ** 2)) * p * r) / (((beta ** 2) * p) + r) if p + r else 0.0
    OC_Score = float((1+beta ** 2) * tp) / OC if OC else 0
    
    f_modified_reciprocal = float(1.0 / f_modified) if f_modified else 0
    OC_Score_reciprocal = float(1.0 / OC_Score) if OC_Score else 0
    f = 1.0 / (float(f_modified_reciprocal) + alpha * OC_Score_reciprocal) if (float(f_modified) + OC_Score_reciprocal) else 0
    # print(f"f_modified: {f_modified}, OC Score: {OC_Score}, F-0.5: {f}")
    # print('=' * 50)
    return round(p, 4), round(r, 4), round(f, 4)

def compute_acc(tp, fp, fn, tn):
    acc = float(tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return round(acc, 4)


class Scorer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, scorer_inputs: List[List[Dict[str, int]]]) -> Dict[str, Any]:
        """ Score evaluation results with the scorer.
        """
        raise NotImplementedError()


@dataclass
class ScorerForGLEU(Scorer):
    order: int = field(
        default=4, metadata={
            "help": "Maximum order of ngrams"
        }
    )
    num_iter: int = field(
        default=500, metadata={
            "help": "Number of iterations to run"
        }
    )
    smoothing: bool = field(
        default=False, metadata={
            "help": "Smoothing factor"
        }
    )

    def __call__(self, scorer_inputs: List[List[Dict]]) -> Dict[str, Any]:
        raise NotImplementedError
