# CSE 572 — Data Mining | Homework 3

**Author:** Brijesh Kumar  
**Course:** CSE 572 — Data Mining  
**University:** Arizona State University  
**Semester:** Spring 2026

---

## Overview

This repository contains the solution for HW3, which covers two major topics:

1. **Task 1 — K-Means Clustering from Scratch**
2. **Task 2 — Recommender Systems**

---

## Repository Structure

```
HW3_CSE572_BrijeshKumar/
│
├── kmeans_scratch.py               # Task 1: K-Means from scratch
├── Recommendation_system_fast.py   # Task 2: Recommender System (SVD, User-CF, Item-CF)
├── generate_report.py              # Generates the PDF report with all plots
│
├── kmeans_data/
│   ├── data.csv                    # MNIST-like image feature vectors (10,000 × 784)
│   ├── label.csv                   # Ground truth class labels (0–9)
│   └── data_description.txt        # Dataset description
│
├── ratings_small.csv               # MovieLens small dataset (100,004 ratings)
│
├── results/
│   ├── kmeans_output.txt           # K-Means terminal output
│   ├── kmeans_summary.csv          # SSE, accuracy, iterations, time per metric
│   ├── sse_vs_iter.png             # SSE convergence curves
│   ├── q4_sse_comparison.png       # Q4: SSE under 3 termination conditions
│   ├── q4_nochange/                # Results: no-centroid-change criterion
│   ├── q4_sse_increase/            # Results: SSE-increase criterion
│   ├── q4_maxiter/                 # Results: max-iterations criterion
│   ├── similarity_comparison.png   # Task 2: similarity metric comparison
│   ├── similarity_results.csv      # Task 2: RMSE/MAE per similarity metric
│   ├── k_sweep.png                 # Task 2: effect of K on RMSE/MAE
│   ├── k_sweep.csv                 # Task 2: K-sweep raw results
│   ├── svd_summary.csv             # SVD/PMF grid search summary
│   ├── final_summary.csv           # All model results combined
│   └── recommender_output.txt      # Recommender terminal output
│
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## Task 1 — K-Means Clustering

### Problem
Implement K-Means clustering **from scratch** (no sklearn KMeans) using three distance metrics:
- **Euclidean distance**
- **1 − Cosine similarity**
- **1 − Generalized Jaccard similarity**

### Key Results (K = 10)

| Metric     | Final SSE    | Accuracy | Iterations | Time (s) |
|------------|-------------|----------|------------|----------|
| Euclidean  | 5.578 × 10⁶ | 46.67%   | 101        | 3.08     |
| Cosine     | 3.072 × 10³ | 54.59%   | 30         | 1.07     |
| **Jaccard**| **3.660 × 10³** | **59.86%** | 31  | 5.87     |

**Jaccard** achieves the best clustering accuracy. **Cosine** converges fastest.

### How to Run
```bash
cd HW3_CSE572_BrijeshKumar
python kmeans_scratch.py
```
Results are saved to `results/`.

---

## Task 2 — Recommender Systems

### Problem
Build a movie recommender system using the MovieLens small dataset and evaluate:
- **SVD / Probabilistic Matrix Factorization (PMF)**
- **User-based Collaborative Filtering (User-CF)**
- **Item-based Collaborative Filtering (Item-CF)**

Evaluation: **5-fold cross-validation**, metrics: **MAE** and **RMSE**.

### Key Results

| Model                  | Avg RMSE | Avg MAE |
|------------------------|----------|---------|
| **SVD / PMF (best)**   | **0.8904** | **0.6860** |
| User-CF (MSD, K=40)    | 0.9678   | 0.7437  |
| Item-CF (MSD, K=40)    | 0.9346   | 0.7210  |
| User-CF (cosine, K=40) | 0.9931   | 0.7671  |
| Item-CF (cosine, K=40) | 0.9951   | 0.7748  |
| User-CF (Pearson, K=40)| 0.9983   | 0.7731  |
| Item-CF (Pearson, K=40)| 0.9891   | 0.7685  |

**SVD/PMF is the best model.** Among CF methods, **Item-CF with MSD** performs best.  
Best K: **40 for User-CF**, **80 for Item-CF**.

### How to Run
```bash
cd HW3_CSE572_BrijeshKumar
python Recommendation_system_fast.py
```
Results are saved to `results/`.

---

## Setup & Installation

```bash
# Create and activate virtual environment
python3 -m venv homew3
source homew3/bin/activate

# Install dependencies
pip install -r requirements.txt

# Note: scikit-surprise requires Cython. On NumPy 2.x you may need to install
# from source with the np.int_t → np.intp_t patch applied.
```

---

## Dependencies

| Package         | Version  |
|-----------------|----------|
| numpy           | 2.4.4    |
| pandas          | 2.3.3    |
| matplotlib      | 3.10.8   |
| scikit-learn    | 1.7.2    |
| scipy           | 1.15.3   |
| scikit-surprise | 1.1.4    |
| reportlab       | 4.4.10   |
| Pillow          | latest   |
