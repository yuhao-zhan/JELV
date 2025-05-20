# JELV: A Judge of Edit-Level Validity

> Judge of Edit-Level Validity for Evaluation and Automated Reference Expansion in Grammatical Error Correction

This repository contains the code and data for the ARR-submitted paper _“JELV: A Judge of Edit-Level Validity for Evaluation and Automated Reference Expansion in Grammatical Error Correction.”_

---

## 🚀 Features

- **PEVData Benchmark**: Human-annotated Pair-wise Edit-Level Validity dataset （PEVData） with 1,459 valid and 1,338 invalid sentence pairs.  
- **Data Expansion & Filtering**: LLM-expanded and JELV-filtered BEA-19 training and development sets in `m2` format.  
- **JELV-CLEME Evaluation**: Edit-level evaluation metric built on CLEME with configurable overcorrection penalty weight `alpha`.

---

## 📁 Repository Structure

```text
.
├── README.md
├── data
│   ├── benchmark
│   └── train
└── evaluation
    ├── JudgeModel
    ├── cleme
    ├── scripts
    └── tests
```

The `evaluation` directory is modified based on [CLEME](https://github.com/THUKElab/CLEME). And more details will be completed after ARR review.

> [!NOTE]
> We will upload the *DeBERTa* checkpoint and more detailed implementations after ARR review.

