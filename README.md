# Project Beta: Fake News Detection System
### High-Throughput Probabilistic Natural Language Processing Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ML Engine](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLTK-WordNet-green.svg)](https://www.nltk.org/)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-96.16%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9919-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Inference-CPU%20Lightweight-lightgrey.svg)]()

---

## 1. Project Overview

**Project Beta (Code: NB-FND-2026)** is a production-grade machine learning system engineered to detect fabricated news articles, clickbait, and coordinated disinformation campaigns. Built entirely on an interpretable probabilistic architecture, the system leverages **WordNet Lemmatization**, **Sublinear TF-IDF Vectorization $(1, 2)$**, and **Complement Naive Bayes (CNB)** classification.

Unlike computationally heavy deep-learning transformers (e.g., BERT, RoBERTa) that require expensive GPU clusters and operate as opaque black boxes, this system:
- Operates entirely on standard **commodity CPU** hardware.
- Trains across **~45,000 articles in under 4 minutes**.
- Delivers **96.16% holdout classification accuracy** with an **ROC-AUC of 0.9919**.
- Provides transparent, legally defensible, feature-level probability evidence suitable for journalistic moderation.
- Exhibits high noise tolerance ($>95.8\%$ accuracy under 10% adversarial typo corruption).

---

## 2. Repository Architecture

```
ml-2-demo/
├── .gitignore                     # Production Git ignore patterns
├── README.md                      # Primary project documentation & quickstart
├── requirements.txt               # Pinned package dependencies
├── data/
│   ├── raw/                       # Original benchmark datasets
│   │   ├── Fake.csv               # 23,481 Fabricated news articles
│   │   └── True.csv               # 21,417 Authentic news articles (Reuters)
│   └── processed/                 # Generated datasets
│       ├── Noisy_Fake.csv         # 10% Adversarial typo-corrupted fake articles
│       └── Noisy_True.csv         # 10% Adversarial typo-corrupted authentic articles
├── docs/                          # Project specifications & technical reports
│   ├── Team2_Fake_News_Detection_PRD_TRD_PD_2.md # Product, Technical & Dev Requirements
│   ├── Final_Project_Report.md    # Summary engineering project report
│   ├── Model_Optimization_and_Scraping_QnA.md    # Architecture review & live scraping analysis
│   ├── audit_log.txt              # Formal verification & data audit trail
│   ├── Project_Report.pdf         # Clean pipeline executive report
│   ├── Noisy_Project_Report.pdf   # Adversarial noise pipeline report
│   └── Comparative_Report.pdf     # Side-by-side comparative PDF report
├── models/                        # Serialized production model artifacts
│   ├── complement_nb.pkl          # Tuned ComplementNB classifier (alpha=0.01)
│   ├── tfidf_vectorizer.pkl       # Fitted 10,000-feature TF-IDF matrix
│   ├── noisy_complement_nb.pkl    # ComplementNB trained on perturbed corpus
│   └── noisy_tfidf_vectorizer.pkl # Vectorizer fitted on perturbed corpus
├── outputs/                       # Publication-grade visual diagnostics (300 DPI)
│   ├── confusion_matrix.png       # Confusion matrix (Clean model)
│   ├── roc_curve.png              # ROC comparison curves (MNB vs CNB vs Tuned)
│   ├── learning_curve.png         # Bias/variance convergence curve
│   ├── wordcloud_fake.png         # Fabricated news linguistic word cloud
│   ├── wordcloud_real.png         # Authentic news linguistic word cloud
│   ├── noisy_confusion_matrix.png # Confusion matrix (Noisy model)
│   ├── noisy_roc_curve.png        # ROC curves (Noisy model)
│   ├── noisy_learning_curve.png   # Learning curve (Noisy model)
│   ├── noisy_wordcloud_fake.png   # Noisy fake linguistic word cloud
│   └── noisy_wordcloud_real.png   # Noisy authentic linguistic word cloud
└── src/                           # Executable source code modules
    ├── __init__.py                # Package initialization
    ├── main_pipeline.py           # Core training, tuning & evaluation pipeline
    ├── generate_noisy_data.py     # Adversarial noise perturbation generator
    ├── noisy_pipeline.py          # Benchmark pipeline on corrupted corpus
    ├── evaluate_robustness.py     # Scientific cross-evaluation stress test
    ├── generate_report.py         # PDF report generator (Clean pipeline)
    ├── generate_noisy_report.py   # PDF report generator (Noisy pipeline)
    └── generate_comparative_report.py # PDF report generator (Comparative)
```

---

## 3. Getting Started

### Prerequisites
- Python 3.10+ (Tested on Python 3.14.6)
- Git

### Installation
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "ml 2 demo"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Corpora:**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

---

## 4. Pipeline Execution Guide

### 1. Execute the Core Machine Learning Pipeline
Trains the baseline MultinomialNB, default ComplementNB, performs 5-fold `GridSearchCV` hyperparameter tuning, evaluates metrics, plots visual figures into `outputs/`, and serializes models to `models/`:
```bash
python src/main_pipeline.py
```

### 2. Generate Adversarial Noisy Datasets (Optional)
Injects 10% controlled typographical perturbations (adjacent character swaps, letter drops, and token omissions) into `data/processed/`:
```bash
python src/generate_noisy_data.py
```

### 3. Execute the Noisy Benchmark Pipeline
Trains and evaluates models on the corrupted dataset:
```bash
python src/noisy_pipeline.py
```

### 4. Run the Robustness Evaluation Harness
Tests the clean model and noisy model head-to-head on the 20% holdout noisy test partition:
```bash
python src/evaluate_robustness.py
```

### 5. Compile PDF Technical Reports
Generates publication-quality PDF summary documents inside `docs/`, including the comprehensive master report:
```bash
# Generate Comprehensive Master Project Report PDF (11 Pages with full tables and visual figures)
python src/generate_master_report_pdf.py

# Generate individual sub-reports
python src/generate_report.py
python src/generate_noisy_report.py
python src/generate_comparative_report.py
```

---

## 5. Performance Benchmarks

### Primary Model Comparison (Unseen Clean Test Partition: 8,978 samples)

| Metric | MultinomialNB (Baseline) | ComplementNB (Default) | Best ComplementNB ($\alpha=0.01$) | Target KPI |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 95.56% | 95.66% | **96.16%** | $\ge 90.0\%$ |
| **Precision (Macro)** | 95.54% | 95.63% | **96.13%** | $\ge 85.0\%$ |
| **Recall (Macro)** | 95.55% | 95.67% | **96.17%** | $\ge 85.0\%$ |
| **F1-Score (Macro)** | 0.9555 | 0.9565 | **0.9615** | $\ge 0.88$ |
| **ROC-AUC** | 0.9897 | 0.9897 | **0.9919** | $\ge 0.92$ |
| **Training Time** | ~3.5 min | ~3.5 min | **~3.8 min** | $< 5.0\text{ min}$ |
| **Model Size** | 388 KB | 400 KB | **400 KB** | $< 15\text{ MB}$ |

### Robustness Stress-Test (Unseen 10% Corrupted Test Partition: 8,978 samples)

| Evaluated Model | Test Condition | Accuracy | Macro F1 | Resilience Retention |
| :--- | :--- | :---: | :---: | :---: |
| **Clean Model** | Clean Test Data | 96.16% | 0.9615 | 100.0% (Ref) |
| **Clean Model** | 10% Corrupted Data | **95.95%** | **0.9594** | **99.78%** |
| **Noisy Model** | 10% Corrupted Data | **95.87%** | **0.9586** | **99.70%** |

---

## 6. Key Technical Decisions

1. **Why Complement Naive Bayes over Multinomial Naive Bayes?**  
   Standard Multinomial Naive Bayes estimates parameters based solely on documents within each class, leading to severe majority-class bias when class frequencies or document lengths differ. Complement Naive Bayes addresses this by computing parameter weights using the *complement* of each class (all documents not in class $c$), significantly improving recall and balance on natural language text.

2. **Why Are There No "Epochs" in Naive Bayes?**  
   Neural networks use epochs because they iteratively update continuous weights using gradient descent and backpropagation. Naive Bayes is a closed-form, probabilistic generative classifier. It calculates joint word frequencies and conditional likelihoods algebraically in a single deterministic pass across the dataset.

3. **Sublinear Term Frequency:**  
   By applying $1 + \log(\text{TF})$, we eliminate the distortion where an article repeating a trigger keyword 30 times overpowers an article that mentions it 2 times in a contextually meaningful way.

---

## 7. Documentation & Reference Links

- [Comprehensive Master Project Report (PDF)](docs/Fake_News_Detection_Comprehensive_Project_Report.pdf): 11-page publication-grade PDF report with full technical analysis, styled tables, and high-resolution visual diagnostics.
- [PRD, TRD & Development Document](docs/Team2_Fake_News_Detection_PRD_TRD_PD_2.md): Complete engineering specification and acceptance criteria.
- [Final Engineering Report](docs/Final_Project_Report.md): Executive summary and visual analysis.
- [Strategic Architecture Q&A](docs/Model_Optimization_and_Scraping_QnA.md): Deep-dive into concept drift, LLM-generated fake news, and live web scraping.
- [Audit Log](docs/audit_log.txt): Verification log of all data splits and hyperparameter choices.

---

## 8. License & Acknowledgments
- **Dataset:** ISOT Fake News Dataset (Kaggle mirror by `emineyetm`), derived from Reuters and PolitiFact.
- **Team:** Team Beta (AI/ML Core Engineering Division)
- **License:** Internal Development Kit — Academic & Operational Evaluation Use.
