# Automated News Analytics Pipeline — NLP Model Benchmarking

Benchmarking study of 12 NLP models across summarization, classification, and topic modeling tasks, evaluated on 300K+ articles from the CNN/DailyMail dataset. The goal was to identify which architectures deliver the best accuracy-to-efficiency tradeoff for large-scale news analytics.

## Overview

News aggregation and analysis pipelines need to summarize articles, classify content, and surface underlying themes at scale. This project benchmarks a range of model architectures — from lightweight to heavyweight — across these three tasks to provide practical, evidence-based guidance for model selection in production NLP systems.

## Tasks & Results

### Summarization
- **Best model:** DistilBART
- **Metric:** ROUGE-L = 24.37
- Lightweight distillation-based model outperformed several larger architectures on this task.

### Classification
- **Best model:** RoBERTa
- **Metric:** Macro F1 = 0.382
- Evaluated across multi-class news categorization.

### Topic Modeling
- **Best configuration:** NMF (Non-negative Matrix Factorization), k=8 topics
- **Metric:** UMass Coherence = −1.647 (least negative among configurations tested)
- Produced the most interpretable and coherent topic clusters compared to alternative topic counts and methods tested.

## Key Takeaway

Bigger models don't always win. Several lighter-weight architectures matched or outperformed heavier models when properly matched to the task — an important consideration for teams optimizing for both accuracy and compute cost in production pipelines.

## Dataset

- **Source:** CNN/DailyMail
- **Scale:** 300,000+ articles

## Models Benchmarked

12 models spanning summarization, classification, and topic modeling approaches, including DistilBART, RoBERTa, and NMF (full list and configurations documented in the accompanying notebooks/scripts).

## Status

This work is currently being developed into a publication.

## Tech Stack

- Python
- Transformers (Hugging Face)
- Text Summarization / Classification pipelines
- Topic Modeling (NMF)
- Large-scale data processing

## Author

Madhu Priya Pulletikurthi
[Portfolio](https://madhupri.github.io) · [GitHub](https://github.com/MaDHuPRi) · [LinkedIn](https://linkedin.com/in/madhu-priya-pulletikurthi)
