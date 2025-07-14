# JELV: A Judge of Edit-Level Validity

> Judge of Edit-Level Validity for Evaluation and Automated Reference Expansion in Grammatical Error Correction

This repository contains the code and data for the under-review paper _“JELV: A Judge of Edit-Level Validity for Evaluation and Automated Reference Expansion in Grammatical Error Correction.”_

## 📁 Repository Structure

```text
.
├── README.md
├── data
│   ├── benchmark
│   └── train
└── evaluation
    ├── FluencyScorer
    ├── JELV_based_cleme
    ├── JELV_based_cleme_cache
    ├── JudgeModel
    ├── demo
    └── scripts
```

---

## 🚀 Features

### Data

We release the two datasets we curated and proposed in our paper.

* **Benchmark**: Human-annotated Pair-wise Edit-Level Validity datase PEVData with 1,459 valid and 1,338 invalid sentence pairs.  
* **Train**: LLM-expanded and JELV-filtered BEA-19's training and development sets in `m2` format, used for re-training top GEC systems in our paper.

### Evaluation

This directory stores the whole workflow of JELV-based $\mathrm{F(x)}$.

* **Base metric**:  [CLEME](https://github.com/THUKElab/CLEME). We choose it becuase our edit-level metric is based on F-score and CLEME is a leading F-metric based metric.
* **Edit-level metric**: combines  JELV-based reclassification, FP decoupling and fluency score integration.
  * **JELV_based_cleme**
    Performs JELV inference on-the-fly: any edit initially flagged as a false positive (FP) during evaluation is sent to JELV for validity checking. This integrates directly into the full JELV-based $\mathrm{F(x)}$ workflow but incurs substantial evaluation overhead.
  * **JELV_based_cleme_cache**
    Uses a precomputed cache of all FP-classified edits (from SEEDA’s meta-evaluation) along with their JELV-validated labels. During evaluation, cached edits bypass inference and immediately return their stored validity—dramatically reducing runtime without compromising accuracy. We employ this "cache" version into our final evaluation workflow.
* **Sentence-level metric**: `FluencyScorer`
* **Final Metric**: JELV-based $\mathrm{F(x)}$, combining edit-level and sentence-level metrics.

---

## ⚒️ How to Use Our Evaluation Metric

* `cd evaluation`

* Requirements

  ```bash
  pip install -r requirements.txt
  ```

* Quick Start

  ```bash
  python scripts/JELV_based_Fx.py --ref demo/ref.m2 --hyp demo/hyp.m2 --alpha 0.5 --gamma 0.5 --level system
  ```
  
  You can configure the hyperparameters according to the optimal settings detailed in Appendix 5.

* Guidelines

  ```bash
  python scripts/JELV_based_Fx.py --help
  ```
  
  This command will output:
  
  ```bash
  usage: JELV_based_Fx.py --ref REF.m2 --hyp HYP.m2 [--alpha α] [--gamma γ] [--level system|sentence]
  
  Compute combined JELV‐based F(x): A + γ·B
  
  options:
    -h, --help            show this help message and exit
    --ref REF             Reference M2 file
    --hyp HYP             Hypothesis M2 file
    --alpha ALPHA         α for the generalized F (default: 0.5)
    --gamma GAMMA         γ weight for fluency term (default: 0.5)
    --level {system,sentence}
                          Evaluation level:
                            system   → single combined score
                            sentence → one combined score per sentence
  ```

> This is an initial github repo. We will further release our model (JELV2.0) checkpoint after review.