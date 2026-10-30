# ACHRP v5 — Automated Hallucination Checking, Retrieval and Repair Pipeline

## Overview

ACHRP v5 is a biomedical answer hallucination detection and repair pipeline. It evaluates whether generated answers are supported by the provided biomedical knowledge and selectively repairs answers identified as hallucinated.

The system combines hybrid information retrieval, Natural Language Inference (NLI), LLM-based judging, machine learning classification, and selective answer repair.

## System Workflow

**Answer → Atomic Claims → Chunking → Hybrid Retrieval → NLI Verification → Feature Extraction → LLM Judge → Classification → Selective Repair → Re-verification**

### Main Components

1. **Claim Extraction:** Breaks answers into smaller, atomic claims.
2. **Chunking:** Divides biomedical knowledge into overlapping three-sentence chunks.
3. **Hybrid Retrieval:** Combines dense semantic retrieval and BM25 keyword retrieval using Reciprocal Rank Fusion (RRF).
4. **NLI Verification:** Checks claims against retrieved chunks, the complete abstract, and individual sentences.
5. **Feature Extraction:** Extracts 18 features, including entailment, contradiction, semantic similarity, and polarity-related features.
6. **LLM Judge:** Uses an LLM to estimate the probability of hallucination.
7. **Classification:** Uses Logistic Regression with grouped nested cross-validation and an internally tuned decision threshold.
8. **Selective Repair:** Repairs detector-flagged answers and verifies the repaired answers again.

## Models and Technologies

- **Answer Generator:** `llama-3.3-70b-versatile`
- **LLM Judge:** `openai/gpt-oss-120b` (through Groq)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **NLI Model:** DeBERTa-v3-large on GPU or DeBERTa-v3-base on CPU, selected automatically.
- **Classifier:** Logistic Regression

### Technologies

- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- Sentence Transformers
- spaCy
- Rank-BM25
- Groq API

## Dataset

The notebook uses a biomedical question-answer dataset containing:

- `Question`
- `Knowledge`
- `Ground Truth`
- `Hallucinated Answer`

Each record provides a question, supporting knowledge, a ground-truth answer, and a hallucinated answer.

The dataset is loaded locally. It is not included in this repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/btwalina/ARCH.git
cd ARCH
```

### 2. Create and activate a virtual environment (optional)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Run the notebook's package installation cell. It checks for and installs missing required packages.

### 4. Configure Groq API keys

Create a `.env` file in the project directory:

```env
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
```

Replace the example values with your own valid API keys.

**Important:** Never upload `.env` or expose API keys publicly.

### 5. Add the dataset

Place the dataset file in the project directory or configure its path in the notebook's configuration cell.

### 6. Run the notebook

Open `achrp_v5_ready.ipynb` in Jupyter Notebook or VS Code and execute the cells sequentially from top to bottom.

The smoke test is optional. The main pipeline uses checkpoints and caches to support resuming interrupted runs.

## Evaluation

The pipeline evaluates hallucination detection using:

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Confusion matrix
- Record-level bootstrap confidence intervals

Grouped nested cross-validation keeps the two answers belonging to the same record together, reducing data leakage between training and testing.

## Output Files

The notebook saves generated results in its output directory, including:

- `achrp_v3_predictions.csv`
- `achrp_v3_claims.csv`
- `achrp_v3_detection_table.csv`
- `achrp_v3_metrics.json`
- Repair evaluation results (when generated)

It also maintains cache files to avoid repeating completed computations and API calls.

## Important Notes

- Run notebook cells in the specified order.
- A valid Groq API key is required for LLM judging, generation, and repair.
- Local embedding and NLI models may require initial downloads.
- GPU availability affects automatic model selection and processing speed.
- Keep datasets, API keys, cache files, and generated outputs private unless you explicitly intend to share them.

## Project Status

ACHRP v5 — Experimental research implementation.

This project explores hybrid retrieval, NLI-based verification, LLM-assisted hallucination detection, and selective correction of biomedical answers.