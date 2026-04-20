"""
HW3 - Task 2: Recommender System with Matrix Data
Author  : Brijesh Kumar
Course  : CSE 572 - Data Mining

Description:
    Builds a movie recommender system using ratings_small.csv.
    Models evaluated:
      - SVD (Probabilistic Matrix Factorization, PMF)
      - User-based Collaborative Filtering (User-CF)
      - Item-based Collaborative Filtering (Item-CF)
    Evaluation: 5-fold cross-validation with MAE and RMSE.
    Also examines the effect of similarity metric and number of neighbours K.

Dataset:
    ratings_small.csv  (MovieLens small, 100K ratings, 671 users, 9066 movies)
    Format: userId, movieId, rating, timestamp
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import KFold, cross_validate, GridSearchCV

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5                           # 5-fold cross-validation

NEIGHBORS = [5, 10, 20, 40, 80]       # K values to sweep for Q2f/g

SVD_PARAM_GRID = {                     # Grid for PMF tuning
    'n_factors': [20, 50, 100],
    'lr_all':    [0.002, 0.005],
    'reg_all':   [0.02,  0.05],
}

K_DEFAULT    = 40                      # Default K for similarity sweep
SIMILARITIES = ['cosine', 'msd', 'pearson']

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(csv_path='ratings_small.csv'):
    """Load ratings CSV and wrap in a Surprise Dataset object."""
    df = pd.read_csv(csv_path)
    required = {'userId', 'movieId', 'rating'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    ratings_df = df[['userId', 'movieId', 'rating']].copy()
    reader     = Reader(rating_scale=(ratings_df['rating'].min(),
                                      ratings_df['rating'].max()))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    return data, ratings_df


def cv_stats(algo, data, name, n_splits=N_SPLITS):
    """
    Run n_splits-fold cross-validation and return mean ± std for RMSE and MAE.
    """
    print(f"\nEvaluating: {name}  ({n_splits}-fold CV) ...", flush=True)
    kf  = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    res = cross_validate(algo, data, measures=['RMSE', 'MAE'],
                         cv=kf, verbose=False, n_jobs=1)
    rmse_m = float(np.array(res['test_rmse']).mean())
    rmse_s = float(np.array(res['test_rmse']).std(ddof=0))
    mae_m  = float(np.array(res['test_mae']).mean())
    mae_s  = float(np.array(res['test_mae']).std(ddof=0))
    print(f"  {name}: RMSE = {rmse_m:.4f} ± {rmse_s:.4f}  "
          f"MAE = {mae_m:.4f} ± {mae_s:.4f}", flush=True)
    return rmse_m, rmse_s, mae_m, mae_s


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_all = time.time()

    print("=" * 60)
    print("  HW3 Task 2 — Recommender Systems")
    print("  Author : Brijesh Kumar")
    print("=" * 60, flush=True)

    data, ratings_df = load_data('ratings_small.csv')
    print(f"\nLoaded ratings : {ratings_df.shape}")
    print(f"CV folds       : {N_SPLITS}\n", flush=True)

    # ── 1. SVD Grid Search (PMF) ──────────────────────────────────────────────
    print("=== Step 1: SVD GridSearchCV (PMF tuning) ===", flush=True)
    t0 = time.time()
    gs = GridSearchCV(SVD, param_grid=SVD_PARAM_GRID,
                      measures=['rmse'], cv=N_SPLITS,
                      n_jobs=-1, joblib_verbose=0)
    gs.fit(data)
    print(f"GridSearch done in {time.time()-t0:.1f}s", flush=True)
    print(f"Best RMSE   : {gs.best_score['rmse']:.4f}")
    print(f"Best params : {gs.best_params['rmse']}", flush=True)

    best_params = gs.best_params['rmse']
    svd_best    = SVD(**best_params, random_state=42)
    svd_rmse_m, svd_rmse_s, svd_mae_m, svd_mae_s = cv_stats(
        svd_best, data, f"SVD/PMF (best) {best_params}")

    gs_df      = pd.DataFrame(gs.cv_results)
    gs_df_path = os.path.join(RESULTS_DIR, "svd_gridsearch_raw.csv")
    gs_df.to_csv(gs_df_path, index=False)
    pd.DataFrame([{
        'model': 'SVD_best', 'params': str(best_params),
        'rmse_mean': svd_rmse_m, 'rmse_std': svd_rmse_s,
        'mae_mean':  svd_mae_m,  'mae_std':  svd_mae_s,
    }]).to_csv(os.path.join(RESULTS_DIR, "svd_summary.csv"), index=False)
    print(f"Saved → {gs_df_path}", flush=True)

    # ── 2. Default SVD baseline ───────────────────────────────────────────────
    print("\n=== Step 2: SVD Default Baseline ===", flush=True)
    d_rmse_m, d_rmse_s, d_mae_m, d_mae_s = cv_stats(
        SVD(random_state=42), data, "SVD/PMF (default)")

    # ── 3. Similarity sweep (User-CF & Item-CF, K=K_DEFAULT) ─────────────────
    print(f"\n=== Step 3: Similarity Metric Sweep (K={K_DEFAULT}) ===", flush=True)
    sim_rows = []
    for sim in SIMILARITIES:
        for user_based, label in [(True, 'User-CF'), (False, 'Item-CF')]:
            algo = KNNBasic(k=K_DEFAULT,
                            sim_options={'name': sim, 'user_based': user_based},
                            verbose=False, n_jobs=-1)
            rm, rs, mm, ms = cv_stats(algo, data, f"{label} ({sim})")
            sim_rows.append({
                'method': 'user' if user_based else 'item',
                'sim': sim, 'k': K_DEFAULT,
                'rmse_mean': rm, 'rmse_std': rs,
                'mae_mean':  mm, 'mae_std':  ms,
            })

    sim_df      = pd.DataFrame(sim_rows)
    sim_df_path = os.path.join(RESULTS_DIR, "similarity_results.csv")
    sim_df.to_csv(sim_df_path, index=False)

    # Similarity plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"HW3 Task 2 — Similarity Metric Impact (K={K_DEFAULT})  |  Brijesh Kumar",
                 fontsize=12, fontweight='bold')
    for ax, metric_key in zip(axes, ['rmse_mean', 'mae_mean']):
        for method, color in [('user', 'royalblue'), ('item', 'tomato')]:
            sub = sim_df[sim_df.method == method]
            ax.errorbar(sub['sim'], sub[metric_key],
                        yerr=sub[metric_key.replace('mean', 'std')],
                        marker='o', label=f"{method}-CF", color=color, linewidth=2)
        ax.set_xlabel("Similarity Metric")
        ax.set_ylabel(metric_key.replace('_mean', '').upper())
        ax.set_title(metric_key.replace('_mean', '').upper())
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sim_plot = os.path.join(RESULTS_DIR, "similarity_comparison.png")
    plt.savefig(sim_plot, dpi=200)
    plt.close()
    print(f"Saved → {sim_df_path}")
    print(f"Saved → {sim_plot}", flush=True)

    # ── 4. K-neighbours sweep (cosine, User-CF & Item-CF) ────────────────────
    print("\n=== Step 4: K-Neighbours Sweep (cosine similarity) ===", flush=True)
    k_rows = []
    for k in NEIGHBORS:
        for user_based, label in [(True, 'User-CF'), (False, 'Item-CF')]:
            algo = KNNBasic(k=k,
                            sim_options={'name': 'cosine', 'user_based': user_based},
                            verbose=False, n_jobs=-1)
            rm, rs, mm, ms = cv_stats(algo, data, f"{label} (cosine) k={k}")
            k_rows.append({
                'method': 'user' if user_based else 'item',
                'sim': 'cosine', 'k': k,
                'rmse_mean': rm, 'rmse_std': rs,
                'mae_mean':  mm, 'mae_std':  ms,
            })

    k_df      = pd.DataFrame(k_rows)
    k_df_path = os.path.join(RESULTS_DIR, "k_sweep.csv")
    k_df.to_csv(k_df_path, index=False)

    # K-sweep plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("HW3 Task 2 — Effect of K on CF Performance  |  Brijesh Kumar",
                 fontsize=12, fontweight='bold')
    for ax, metric_key in zip(axes, ['rmse_mean', 'mae_mean']):
        for method, color in [('user', 'royalblue'), ('item', 'tomato')]:
            sub = k_df[k_df.method == method]
            ax.errorbar(sub['k'], sub[metric_key],
                        yerr=sub[metric_key.replace('mean', 'std')],
                        marker='o', label=f"{method}-CF", color=color, linewidth=2)
        ax.set_xlabel("K (number of neighbours)")
        ax.set_ylabel(metric_key.replace('_mean', '').upper())
        ax.set_title(f"{metric_key.replace('_mean','').upper()} vs K")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    k_plot = os.path.join(RESULTS_DIR, "k_sweep.png")
    plt.savefig(k_plot, dpi=200)
    plt.close()
    print(f"Saved → {k_df_path}")
    print(f"Saved → {k_plot}", flush=True)

    # Best K per method
    for method in ['user', 'item']:
        sub    = k_df[k_df.method == method]
        best_k = int(sub.loc[sub['rmse_mean'].idxmin(), 'k'])
        best_r = sub['rmse_mean'].min()
        print(f"  Best K for {method}-CF : K={best_k}  (RMSE={best_r:.4f})")

    # ── 5. Final summary table ────────────────────────────────────────────────
    best_neigh = sim_df.loc[sim_df['rmse_mean'].idxmin()]
    final = pd.DataFrame([
        {
            'model':     'SVD_best (PMF)',
            'params':    str(best_params),
            'rmse_mean': svd_rmse_m, 'rmse_std': svd_rmse_s,
            'mae_mean':  svd_mae_m,  'mae_std':  svd_mae_s,
        },
        {
            'model':     f"best_CF_{best_neigh['method']}_{best_neigh['sim']}",
            'params':    f"k={int(best_neigh['k'])}",
            'rmse_mean': float(best_neigh['rmse_mean']),
            'rmse_std':  float(best_neigh['rmse_std']),
            'mae_mean':  float(best_neigh['mae_mean']),
            'mae_std':   float(best_neigh['mae_std']),
        },
    ])
    final_path = os.path.join(RESULTS_DIR, "final_summary.csv")
    final.to_csv(final_path, index=False)
    print(f"\nSaved → {final_path}")
    print("\nFinal Summary:")
    print(final.to_string(index=False))

    print(f"\nAll experiments completed in {time.time()-start_all:.1f}s", flush=True)
    print("Results saved in:", RESULTS_DIR)
    print("Files:", os.listdir(RESULTS_DIR))


if __name__ == "__main__":
    main()
