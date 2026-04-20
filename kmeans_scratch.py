#!/usr/bin/env python3
"""
HW3 - Task 1: K-Means Clustering from Scratch
Author  : Brijesh Kumar
Course  : CSE 572 - Data Mining

Description:
    Implements K-Means clustering from scratch using three distance metrics:
      - Euclidean distance
      - 1 - Cosine similarity
      - 1 - Generalized Jaccard similarity
    Answers Q1 (SSE), Q2 (Accuracy), Q3 (Convergence), Q4 (Stop Conditions).

Usage:
    python3 kmeans_scratch.py --n_init 10 --max_iters 500 --stop_rule both

Outputs:
    results/kmeans_summary.csv         : summary table per metric (best run)
    results/sse_history_{metric}.npy   : SSE history saved as numpy array
    results/sse_vs_iter.png            : SSE convergence plot
"""

import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Numeric epsilon ────────────────────────────────────────────────────────────
_eps = 1e-12

# ══════════════════════════════════════════════════════════════════════════════
# Distance matrix helpers
# ══════════════════════════════════════════════════════════════════════════════

def euclidean_distance_matrix(X, C):
    """
    Euclidean distance between every row of X and every row of C.
    X : (n, d)  C : (K, d)  →  returns (n, K)
    Uses the identity ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c for speed.
    """
    XX = np.sum(X * X, axis=1, keepdims=True)   # (n, 1)
    CC = np.sum(C * C, axis=1, keepdims=True).T  # (1, K)
    XC = X.dot(C.T)                              # (n, K)
    d2 = XX + CC - 2.0 * XC
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


def cosine_distance_matrix(X, C):
    """
    1 - cosine_similarity.
    X : (n, d)  C : (K, d)  →  returns (n, K)
    """
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    C_norm = np.linalg.norm(C, axis=1, keepdims=True).T
    denom  = np.maximum(X_norm * C_norm, _eps)
    sim    = np.clip(X.dot(C.T) / denom, -1.0, 1.0)
    return 1.0 - sim


def jaccard_distance_matrix(X, C):
    """
    1 - Generalized Jaccard similarity for non-negative vectors.
    J(x, c) = sum(min(x_i, c_i)) / sum(max(x_i, c_i))
    X : (n, d)  C : (K, d)  →  returns (n, K)
    """
    n  = X.shape[0]
    Kc = C.shape[0]
    numer = np.empty((n, Kc), dtype=np.float64)
    denom = np.empty((n, Kc), dtype=np.float64)
    for k in range(Kc):
        c           = C[k]
        numer[:, k] = np.minimum(X, c).sum(axis=1)
        denom[:, k] = np.maximum(np.maximum(X, c).sum(axis=1), _eps)
    return 1.0 - numer / denom


# ══════════════════════════════════════════════════════════════════════════════
# K-Means++ initialisation
# ══════════════════════════════════════════════════════════════════════════════

def kmeans_pp_init(X, K, rng):
    """
    K-Means++ seeding: spread initial centroids using distance-weighted sampling.
    """
    n       = X.shape[0]
    centers = np.empty((K, X.shape[1]), dtype=np.float64)
    centers[0] = X[int(rng.integers(low=0, high=n))]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        if d2.sum() <= 0:
            idx = int(rng.integers(low=0, high=n))
        else:
            idx = int(rng.choice(n, p=d2 / d2.sum()))
        centers[k] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centers[k]) ** 2, axis=1))
    return centers


# ══════════════════════════════════════════════════════════════════════════════
# KMeans class
# ══════════════════════════════════════════════════════════════════════════════

class KMeansScratch:
    """
    K-Means from scratch supporting Euclidean, Cosine, and Jaccard distances.

    Parameters
    ----------
    K         : number of clusters
    metric    : 'euclidean' | 'cosine' | 'jaccard'
    max_iters : maximum number of iterations
    tol       : centroid-shift tolerance for 'nochange' stop rule
    rng_seed  : random seed for reproducibility
    init      : 'kmeans++' or 'random'
    stop_rule : 'both' | 'nochange' | 'sse_increase' | 'maxiter'
                'both' = stop when no centroid change OR SSE increases
    """

    def __init__(self, K, metric='euclidean', max_iters=500, tol=1e-6,
                 rng_seed=42, init='kmeans++', stop_rule='both'):
        self.K         = int(K)
        self.metric    = metric
        self.max_iters = int(max_iters)
        self.tol       = float(tol)
        self.rng       = np.random.default_rng(rng_seed)
        self.init      = init
        self.stop_rule = stop_rule
        self.centroids  = None
        self.SSE_history = []
        self.iterations  = 0

    def _distance_matrix(self, X, C):
        if self.metric == 'euclidean':
            return euclidean_distance_matrix(X, C)
        elif self.metric == 'cosine':
            return cosine_distance_matrix(X, C)
        elif self.metric == 'jaccard':
            return jaccard_distance_matrix(X, C)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X, verbose=False):
        """
        Fit K-Means on data matrix X.

        Returns
        -------
        labels   : (n,) cluster assignment per sample
        SSE      : final Sum of Squared Errors
        iters    : number of iterations run
        runtime  : wall-clock time in seconds
        """
        X = X.astype(np.float64)
        n, _ = X.shape

        # Initialise centroids
        if self.init == 'kmeans++':
            C = kmeans_pp_init(X, self.K, self.rng)
        else:
            idx = self.rng.choice(n, size=self.K, replace=False)
            C   = X[idx].astype(np.float64)

        prev_SSE       = np.inf
        self.SSE_history = []
        self.iterations  = 0
        start_time     = time.time()

        for it in range(1, self.max_iters + 1):
            # Assignment step
            D      = self._distance_matrix(X, C)
            labels = np.argmin(D, axis=1)
            min_d  = D[np.arange(n), labels]
            SSE    = float(np.sum(min_d ** 2))
            self.SSE_history.append(SSE)

            # Update step
            C_new = np.zeros_like(C)
            for k in range(self.K):
                members = X[labels == k]
                if len(members) == 0:
                    C_new[k] = X[int(self.rng.integers(n))]
                else:
                    C_new[k] = members.mean(axis=0)

            # Guard against NaN / Inf centroids
            bad = ~np.isfinite(C_new).all(axis=1)
            for k in np.where(bad)[0]:
                C_new[k] = X[int(self.rng.integers(n))]

            # Evaluate stopping conditions
            shift          = np.linalg.norm(C_new - C, axis=1)
            no_change      = bool(np.all(shift <= self.tol))
            sse_increased  = SSE > prev_SSE + 1e-12

            C        = C_new
            prev_SSE = SSE
            self.iterations = it

            if verbose:
                print(f"[{self.metric}] iter {it:3d}  SSE {SSE:.6e}  "
                      f"shift_max {shift.max():.3e}  sse_inc {sse_increased}")

            if self.stop_rule == 'both':
                if no_change or sse_increased:
                    break
            elif self.stop_rule == 'nochange':
                if no_change:
                    break
            elif self.stop_rule == 'sse_increase':
                if sse_increased:
                    break
            elif self.stop_rule == 'maxiter':
                pass
            else:
                if no_change or sse_increased:
                    break

        self.centroids = C
        return labels, SSE, int(self.iterations), float(time.time() - start_time)

    def predict(self, X):
        D = self._distance_matrix(X, self.centroids)
        return np.argmin(D, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def cluster_label_map(y_true, labels, K):
    """Majority-vote label per cluster."""
    mapping = {}
    for k in range(K):
        members    = y_true[labels == k]
        mapping[k] = Counter(members).most_common(1)[0][0] if len(members) else None
    return mapping


def labels_to_predictions(labels, mapping):
    return np.array([mapping[l] if mapping[l] is not None else -1 for l in labels])


def accuracy_from_labels(y_true, preds):
    valid = preds != -1
    if valid.sum() == 0:
        return 0.0
    return float((preds[valid] == y_true[valid]).mean())


def run_with_n_init(X, K, metric, n_init=10, rng_seed=42, **km_kwargs):
    """Run K-Means n_init times and return the best (lowest SSE) result."""
    best_sse   = None
    best_tuple = None
    for i in range(n_init):
        km = KMeansScratch(K=K, metric=metric, rng_seed=rng_seed + i, **km_kwargs)
        labels, sse, iters, runtime = km.fit(X)
        if best_sse is None or sse < best_sse:
            best_sse   = sse
            best_tuple = (km, labels, sse, iters, runtime)
    return best_tuple


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("  HW3 Task 1 — K-Means Clustering")
    print("  Author : Brijesh Kumar")
    print("=" * 60)

    # Load data
    X = pd.read_csv('kmeans_data/data.csv').values.astype(np.float64)
    y = pd.read_csv('kmeans_data/label.csv').values.flatten()
    K = int(np.unique(y).shape[0])
    print(f"\nData shape : {X.shape}   K = {K}")
    print(f"Pixel range: [{X.min():.1f}, {X.max():.1f}]  "
          f"mean={X.mean():.2f}  std={X.std():.2f}\n")

    # Preprocessing for each metric
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)
    X_cos  = normalize(X_std, axis=1)

    X_j        = X_std.copy()
    min_val    = X_j.min()
    if min_val < 0:
        X_j -= min_val
    feat_range = X_j.max(axis=0) - X_j.min(axis=0)
    feat_range[feat_range == 0] = 1.0
    X_j = (X_j - X_j.min(axis=0)) / feat_range

    data_variants = {
        'euclidean': X_std,
        'cosine':    X_cos,
        'jaccard':   X_j,
    }

    metrics      = ['euclidean', 'cosine', 'jaccard']
    summary_rows = []

    for metric in metrics:
        print(f"=== Running metric: {metric} ===")
        X_run = data_variants[metric].astype(np.float64)

        model, labels, sse, iters, runtime = run_with_n_init(
            X_run, K, metric,
            n_init    = args.n_init,
            rng_seed  = args.seed,
            max_iters = args.max_iters,
            tol       = args.tol,
            init      = 'kmeans++',
            stop_rule = args.stop_rule,
        )

        mapping = cluster_label_map(y, labels, K)
        preds   = labels_to_predictions(labels, mapping)
        acc     = accuracy_from_labels(y, preds)

        hist_path = os.path.join(args.results_dir, f"sse_history_{metric}.npy")
        np.save(hist_path, np.array(model.SSE_history, dtype=np.float64))

        row = {
            'metric':    metric,
            'final_SSE': float(sse),
            'iters':     int(iters),
            'time_s':    float(runtime),
            'accuracy':  float(acc),
        }
        summary_rows.append(row)
        print(f"  SSE={sse:.6e}  iters={iters}  time={runtime:.3f}s  acc={acc:.4f}")
        print(f"  Cluster→label map: {dict(list(mapping.items())[:5])} ...\n")

    # Save summary CSV
    df       = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.results_dir, "kmeans_summary.csv")
    df.to_csv(csv_path, index=False)
    print("\nFull Results Table:")
    print(df.to_string(index=False))
    print(f"\nSaved → {csv_path}")

    # SSE convergence plot
    plt.figure(figsize=(9, 5))
    colors = {'euclidean': 'steelblue', 'cosine': 'darkorange', 'jaccard': 'mediumseagreen'}
    for m in metrics:
        p = os.path.join(args.results_dir, f"sse_history_{m}.npy")
        if os.path.exists(p):
            hist = np.load(p)
            plt.plot(np.arange(1, len(hist) + 1), hist,
                     label=m, color=colors[m], linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.yscale('log')
    plt.title("HW3 Task 1 — SSE Convergence  |  Brijesh Kumar")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, "sse_vs_iter.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved → {plot_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="HW3 Task 1 – K-Means from scratch (Brijesh Kumar)"
    )
    p.add_argument("--n_init",      type=int,   default=10)
    p.add_argument("--max_iters",   type=int,   default=500)
    p.add_argument("--tol",         type=float, default=1e-6)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--stop_rule",   type=str,   default="both",
                   choices=['both', 'nochange', 'sse_increase', 'maxiter'])
    p.add_argument("--results_dir", type=str,   default="results")
    args = p.parse_args()
    main(args)
