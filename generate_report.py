"""
generate_report.py
HW3 PDF Report Generator — Brijesh Kumar
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
PDF_OUT = os.path.join(BASE, "Brijesh_Kumar_HW3_Report.pdf")

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE    = colors.HexColor("#1a4fa0")
LBLUE   = colors.HexColor("#dce8ff")
DBLUE   = colors.HexColor("#0d2a5e")
GRAY    = colors.HexColor("#f5f5f5")
DGRAY   = colors.HexColor("#555555")
GREEN   = colors.HexColor("#1e7e34")
RED     = colors.HexColor("#c0392b")
WHITE   = colors.white

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE   = S("Title2",   fontSize=22, textColor=WHITE,    alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
SUBTITLE= S("Sub",      fontSize=12, textColor=LBLUE,    alignment=TA_CENTER, fontName="Helvetica",      spaceAfter=2)
SECTION = S("Section",  fontSize=14, textColor=WHITE,    alignment=TA_LEFT,   fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=4, leftIndent=6)
QHEAD   = S("QHead",    fontSize=12, textColor=DBLUE,    fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2)
BODY    = S("Body2",    fontSize=10, textColor=colors.black, fontName="Helvetica",  spaceBefore=2,  spaceAfter=2, leading=14, alignment=TA_JUSTIFY)
BULLET  = S("Bullet2",  fontSize=10, textColor=colors.black, fontName="Helvetica",  spaceBefore=1,  spaceAfter=1, leading=13, leftIndent=16, bulletIndent=6)
CAPTION = S("Caption",  fontSize=9,  textColor=DGRAY,    fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceBefore=1, spaceAfter=4)
NOTE    = S("Note",     fontSize=9,  textColor=DGRAY,    fontName="Helvetica-Oblique", spaceBefore=1, spaceAfter=3)
FOOTER  = S("Footer",   fontSize=8,  textColor=DGRAY,    alignment=TA_CENTER, fontName="Helvetica")

# ── Helper builders ───────────────────────────────────────────────────────────

def section_banner(text):
    data = [[Paragraph(text, SECTION)]]
    t = Table(data, colWidths=[7.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BLUE),
        ("ROUNDEDCORNERS", [4]),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    return t

def q_header(text):
    return Paragraph(text, QHEAD)

def body(text):
    return Paragraph(text, BODY)

def bullet(text):
    return Paragraph(f"• {text}", BULLET)

def caption(text):
    return Paragraph(f"<i>{text}</i>", CAPTION)

def note(text):
    return Paragraph(f"<i>Note: {text}</i>", NOTE)

def img(path, width=6.5*inch):
    if os.path.exists(path):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, h = im.size
        aspect = h / w
        return Image(path, width=width, height=width * aspect)
    return body(f"[Image not found: {path}]")

def spacer(h=0.08):
    return Spacer(1, h * inch)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6)

def styled_table(headers, rows, col_widths=None, highlight_last=False):
    data = [headers] + rows
    if col_widths is None:
        col_widths = [7.0 * inch / len(headers)] * len(headers)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",   (0, 0), (-1,  0), BLUE),
        ("TEXTCOLOR",    (0, 0), (-1,  0), WHITE),
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, GRAY]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]
    if highlight_last:
        style.append(("BACKGROUND", (0, len(rows)), (-1, len(rows)), colors.HexColor("#e8f5e9")))
        style.append(("FONTNAME",   (0, len(rows)), (-1, len(rows)), "Helvetica-Bold"))
        style.append(("TEXTCOLOR",  (0, len(rows)), (-1, len(rows)), GREEN))
    t.setStyle(TableStyle(style))
    return t

# ── Generate extra plots ──────────────────────────────────────────────────────

def make_q1_q2_q3_plot():
    metrics   = ["Euclidean", "Cosine", "Jaccard"]
    sses      = [5.578e6,     3.072e3,  3.660e3]
    accs      = [0.4667,      0.5459,   0.5986]
    iters     = [101,         30,       31]
    times     = [3.08,        1.07,     5.87]
    colors_   = ["#3b82f6",   "#f97316","#22c55e"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Task 1 — K-Means Summary (Q1, Q2, Q3)  |  Brijesh Kumar",
                 fontsize=14, fontweight="bold", y=1.01)

    titles = ["Q1 — Final SSE", "Q2 — Clustering Accuracy", "Q3 — Iterations", "Q3 — Time (s)"]
    vals   = [sses, accs, iters, times]
    ylabs  = ["SSE", "Accuracy", "Iterations", "Time (seconds)"]

    for ax, title, v, ylab, c in zip(axes.flat, titles, vals, ylabs,
                                      [colors_]*4):
        bars = ax.bar(metrics, v, color=colors_, alpha=0.88, edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, v):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    f"{val:.2e}" if title=="Q1 — Final SSE" else f"{val:.4f}" if "Accuracy" in title else f"{val}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RESULTS, "q1_q2_q3_summary.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def make_q4_plot():
    stops   = ["No Centroid\nChange", "SSE\nIncrease", "Max Iter\n(100)"]
    data    = {
        "Euclidean": [5.578e6,  5.578e6,  5.578e6],
        "Cosine":    [3.059e3,  3.072e3,  3.059e3],
        "Jaccard":   [3.660e3,  3.660e3,  3.660e3],
    }
    colors_ = {"Euclidean": "#3b82f6", "Cosine": "#f97316", "Jaccard": "#22c55e"}
    x       = np.arange(len(stops))
    width   = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Task 1 Q4 — Final SSE Under Three Termination Conditions  |  Brijesh Kumar",
                 fontsize=13, fontweight="bold")

    for idx, (metric, vals) in enumerate(data.items()):
        ax   = axes[idx]
        bars = ax.bar(stops, vals, color=colors_[metric], alpha=0.88, width=0.5,
                      edgecolor="white", linewidth=1.2)
        ax.set_title(f"{metric} K-Means", fontsize=12, fontweight="bold")
        ax.set_ylabel("Final SSE"); ax.set_ylim(0, max(vals)*1.2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v*1.02,
                    f"{v:.2e}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RESULTS, "q4_final.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def make_task2_model_comparison():
    models = ["SVD/PMF", "User-CF\n(cosine)", "Item-CF\n(cosine)",
              "User-CF\n(MSD)", "Item-CF\n(MSD)", "User-CF\n(Pearson)", "Item-CF\n(Pearson)"]
    rmses  = [0.8904, 0.9931, 0.9951, 0.9678, 0.9346, 0.9983, 0.9891]
    maes   = [0.6860, 0.7671, 0.7748, 0.7437, 0.7210, 0.7731, 0.7685]
    cols   = ["#3b82f6" if i==0 else "#94a3b8" for i in range(len(models))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task 2 — Model Comparison (5-fold CV)  |  Brijesh Kumar",
                 fontsize=13, fontweight="bold")

    for ax, vals, metric in zip(axes, [rmses, maes], ["RMSE", "MAE"]):
        bars = ax.bar(models, vals, color=cols, alpha=0.88, edgecolor="white", linewidth=1.2)
        ax.set_title(f"Average {metric}", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric); ax.set_ylim(0, max(vals)*1.15)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        # Annotate best
        best_idx = np.argmin(vals)
        bars[best_idx].set_edgecolor("#1e7e34"); bars[best_idx].set_linewidth(2.5)

    legend = [mpatches.Patch(color="#3b82f6", label="SVD/PMF (best)"),
              mpatches.Patch(color="#94a3b8", label="CF models")]
    axes[0].legend(handles=legend, fontsize=9)
    plt.tight_layout()
    out = os.path.join(RESULTS, "task2_full_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf():
    print("Generating PDF…")

    # Pre-generate plots
    q1q2q3_plot = make_q1_q2_q3_plot()
    q4_plot     = make_q4_plot()
    t2_plot     = make_task2_model_comparison()
    sse_plot    = os.path.join(RESULTS, "sse_vs_iter.png")
    sim_plot    = os.path.join(RESULTS, "similarity_comparison.png")
    k_plot      = os.path.join(RESULTS, "k_sweep.png")

    doc = SimpleDocTemplate(
        PDF_OUT,
        pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
    )

    story = []

    # ── Cover banner ──────────────────────────────────────────────────────────
    cover_data = [[
        Paragraph("CSE 572 — Data Mining",          SUBTITLE),
    ],[
        Paragraph("Homework 3 Report",               TITLE),
    ],[
        Paragraph("Brijesh Kumar",                   SUBTITLE),
    ],[
        Paragraph("Arizona State University  •  Spring 2026", SUBTITLE),
    ]]
    cover_tbl = Table(cover_data, colWidths=[7.0*inch])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), DBLUE),
        ("TOPPADDING",   (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",  (0,0),(-1,-1), 20),
        ("RIGHTPADDING", (0,0),(-1,-1), 20),
    ]))
    story += [cover_tbl, spacer(0.2)]

    # ── Dataset overview ──────────────────────────────────────────────────────
    story += [
        body("<b>Dataset Overview:</b> Task 1 uses a 10,000-sample MNIST-like image dataset "
             "(784 pixel features, 10 classes). Task 2 uses the MovieLens small dataset "
             "(100,004 ratings, 671 users, 9,066 movies, ratings 0.5–5.0)."),
        spacer(0.05),
        hr(),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # TASK 1
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_banner("TASK 1 — K-Means Clustering from Scratch"), spacer(0.08)]

    story += [
        body("K-Means was implemented from scratch using three distance metrics: "
             "<b>Euclidean distance</b>, <b>1 − Cosine similarity</b>, and "
             "<b>1 − Generalized Jaccard similarity</b>.  K-Means++ initialization "
             "was used with 5 random restarts; the run with the lowest SSE was selected. "
             "K = 10 (number of unique class labels)."),
    ]

    # ── Q1 ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q1 — SSE Comparison  (10 pts)"),
        body("The table and chart below show the final Sum of Squared Errors (SSE) for each metric. "
             "Note: Euclidean SSE is naturally larger in magnitude because it operates on "
             "standardized 784-dimensional vectors; Cosine and Jaccard SSEs are computed "
             "on their respective normalized spaces."),
        spacer(0.04),
        styled_table(
            ["Metric", "Final SSE", "Normalized Scale"],
            [["Euclidean", "5.578 × 10⁶", "Large (magnitude-based)"],
             ["Cosine",    "3.072 × 10³", "Small (angle-based)"],
             ["Jaccard",   "3.660 × 10³", "Small (overlap-based)"]],
            col_widths=[2.3*inch, 2.3*inch, 2.4*inch],
        ),
        spacer(0.04),
        body("<b>Conclusion:</b> Jaccard achieves the best (most compact) cluster structure "
             "in its normalized feature space. Cosine is second-best. Euclidean, while "
             "showing the largest absolute SSE, suffers from the curse of dimensionality "
             "in 784-D space."),
    ]

    # ── Q2 ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q2 — Clustering Accuracy via Majority Vote  (10 pts)"),
        body("Each cluster was labeled using the majority class of its members. "
             "Predicted labels were compared against true labels to compute accuracy."),
        spacer(0.04),
        styled_table(
            ["Metric", "Accuracy", "Rank"],
            [["Euclidean", "0.4667", "3rd"],
             ["Cosine",    "0.5459", "2nd"],
             ["Jaccard",   "0.5986", "1st ✓"]],
            col_widths=[2.3*inch, 2.3*inch, 2.4*inch],
            highlight_last=True,
        ),
        spacer(0.04),
        body("<b>Conclusion:</b> <b>Jaccard</b> achieves the highest accuracy (~0.60). "
             "Its overlap-based similarity captures pixel intensity patterns better than "
             "magnitude-based (Euclidean) or pure direction-based (Cosine) metrics on "
             "non-negative image data."),
    ]

    # ── Q1/Q2/Q3 combined plot ────────────────────────────────────────────────
    story += [
        spacer(0.04),
        img(q1q2q3_plot, width=7.0*inch),
        caption("Figure 1 — Summary of SSE, Accuracy, Iterations, and Time for all three metrics."),
    ]

    # ── Q3 ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q3 — Convergence Under Combined Stop Criteria  (10 pts)"),
        body("Stop criteria used: <i>no centroid change</i> <b>OR</b> <i>SSE increases</i> "
             "<b>OR</b> <i>maximum 500 iterations reached</i>."),
        spacer(0.04),
        styled_table(
            ["Metric", "Iterations", "Time (s)", "Accuracy"],
            [["Euclidean", "101",  "3.08",  "0.4667"],
             ["Cosine",    "30",   "1.07",  "0.5459"],
             ["Jaccard",   "31",   "5.87",  "0.5986"]],
            col_widths=[1.75*inch, 1.75*inch, 1.75*inch, 1.75*inch],
        ),
        spacer(0.04),
        body("<b>Conclusion:</b>"),
        bullet("<b>Most iterations:</b> Euclidean (101) — slowest to converge because "
               "arithmetic-mean centroid updates are not aligned with the Euclidean gradient in "
               "high-dimensional standardized space."),
        bullet("<b>Most wall-clock time:</b> Jaccard (5.87s) — fewer iterations but each "
               "iteration is heavier due to pairwise min/max operations over 784 features."),
        bullet("<b>Fastest overall:</b> Cosine — vectorized dot-product operations converge "
               "in just 30 iterations with minimal per-iteration cost."),
        spacer(0.04),
        img(sse_plot, width=7.0*inch),
        caption("Figure 2 — SSE convergence curves for all three metrics (log scale)."),
    ]

    story.append(PageBreak())

    # ── Q4 ────────────────────────────────────────────────────────────────────
    story += [
        section_banner("TASK 1 (continued) — Q4 & Q5"),
        spacer(0.08),
        q_header("Q4 — SSE Under Three Separate Termination Conditions  (10 pts)"),
        body("K-Means was run three separate times, each with only one termination condition active:"),
        spacer(0.04),
        styled_table(
            ["Metric", "No Centroid Change", "SSE Increase", "Max Iter = 100"],
            [["Euclidean", "5.578 × 10⁶", "5.578 × 10⁶", "5.578 × 10⁶"],
             ["Cosine",    "3.059 × 10³", "3.072 × 10³", "3.059 × 10³"],
             ["Jaccard",   "3.660 × 10³", "3.660 × 10³", "3.660 × 10³"]],
            col_widths=[1.6*inch, 1.8*inch, 1.8*inch, 1.8*inch],
        ),
        spacer(0.04),
        body("<b>Analysis:</b>"),
        bullet("<b>No Centroid Change:</b> Runs until full convergence — produces the lowest "
               "achievable SSE for each metric. Cosine needs 123 iterations to satisfy this "
               "strict criterion."),
        bullet("<b>SSE Increase:</b> Stops as soon as SSE ticks upward. Cosine stops at "
               "iteration 30 (one step before centroid stabilisation), giving a slightly "
               "higher SSE (3.072 × 10³ vs 3.059 × 10³). Euclidean and Jaccard are "
               "unaffected — their SSE never increases before convergence."),
        bullet("<b>Max Iter = 100:</b> Euclidean converges at iteration 101 — cutting off "
               "at 100 does not change the final SSE because convergence had essentially "
               "been reached. Cosine and Jaccard converge well before iteration 100 "
               "and are therefore unaffected."),
        bullet("<b>Key insight:</b> All three stop conditions yield identical SSEs for "
               "Euclidean and Jaccard. For Cosine, the 'SSE Increase' rule stops one "
               "step early and gives a marginally worse result."),
        spacer(0.04),
        img(q4_plot, width=7.0*inch),
        caption("Figure 3 — Final SSE comparison under three separate termination conditions."),
    ]

    # ── Q5 ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q5 — Summary Observations  (5 pts)"),
        bullet("<b>Distance metric matters greatly</b> in K-Means. Preprocessing (standardisation "
               "for Euclidean, L2-normalisation for Cosine, [0,1] scaling for Jaccard) is "
               "critical for fair comparison."),
        bullet("<b>Jaccard yields the best cluster purity (≈ 0.60 accuracy)</b> on this MNIST-like "
               "pixel dataset because its overlap-based similarity naturally handles "
               "non-negative, sparse intensity values."),
        bullet("<b>Cosine is the most computationally efficient</b> — converges in 30 iterations "
               "using fast vectorised dot-products, making it the best speed–accuracy trade-off."),
        bullet("<b>Euclidean performs worst</b> in 784-D space. Magnitude differences across "
               "hundreds of features inflate distances and slow convergence (101 iterations)."),
        bullet("<b>Stop condition choice</b>: the 'no centroid change' rule guarantees the lowest "
               "SSE but takes the most iterations. 'SSE increase' can terminate prematurely. "
               "'Max iter' is safe when the limit exceeds natural convergence."),
        bullet("<b>K-Means++ initialisation</b> consistently reduces sensitivity to random seeds "
               "and leads to faster convergence compared to random initialisation."),
        spacer(0.04),
        hr(),
    ]

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # TASK 2
    # ══════════════════════════════════════════════════════════════════════════
    story += [section_banner("TASK 2 — Recommender Systems (Matrix Data)"), spacer(0.08)]

    story += [
        body("A movie recommender system was built using the MovieLens small dataset "
             "(<b>ratings_small.csv</b> — 100,004 ratings, 671 users, 9,066 movies, "
             "scale 0.5–5.0). Three models were evaluated: "
             "SVD (Probabilistic Matrix Factorization / PMF), "
             "User-based Collaborative Filtering (User-CF), and "
             "Item-based Collaborative Filtering (Item-CF) using the "
             "<b>scikit-surprise</b> library with <b>5-fold cross-validation</b>."),
    ]

    # ── 2c ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q2c — Average MAE and RMSE (5-fold CV)  (10 pts)"),
        body("All models were evaluated using 5-fold cross-validation. "
             "MAE (Mean Absolute Error) measures average prediction error; "
             "RMSE (Root Mean Squared Error) penalises large errors more heavily."),
        spacer(0.04),
        styled_table(
            ["Model", "Avg RMSE", "Avg MAE"],
            [["SVD / PMF (best params)", "0.8904 ± 0.0041", "0.6860 ± 0.0034"],
             ["SVD / PMF (default)",     "0.8969 ± 0.0037", "0.6909 ± 0.0031"],
             ["User-CF (cosine, K=40)",  "0.9931 ± 0.0025", "0.7671 ± 0.0032"],
             ["Item-CF (cosine, K=40)",  "0.9951 ± 0.0064", "0.7748 ± 0.0052"],
             ["User-CF (MSD, K=40)",     "0.9678 ± 0.0024", "0.7437 ± 0.0032"],
             ["Item-CF (MSD, K=40)",     "0.9346 ± 0.0059", "0.7210 ± 0.0052"],
             ["User-CF (Pearson, K=40)", "0.9983 ± 0.0027", "0.7731 ± 0.0033"],
             ["Item-CF (Pearson, K=40)", "0.9891 ± 0.0063", "0.7685 ± 0.0043"]],
            col_widths=[3.0*inch, 2.0*inch, 2.0*inch],
        ),
    ]

    # ── 2d ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q2d — Best Model Comparison  (10 pts)"),
        body("The chart below compares the three primary model types. "
             "SVD/PMF is highlighted as the top performer."),
        spacer(0.04),
        img(t2_plot, width=7.0*inch),
        caption("Figure 4 — Average RMSE and MAE for all models (5-fold CV). "
                "SVD/PMF (blue) is best; green border marks the minimum."),
        spacer(0.04),
        body("<b>Conclusion:</b> <b>SVD/PMF is the best model</b> (RMSE = 0.8904, MAE = 0.6860). "
             "It outperforms both CF approaches because latent-factor models capture global "
             "user–item interaction patterns beyond simple pairwise similarity. Among "
             "neighbourhood methods, Item-CF with MSD performs best (RMSE = 0.9346)."),
    ]

    story.append(PageBreak())

    # ── 2e ────────────────────────────────────────────────────────────────────
    story += [
        section_banner("TASK 2 (continued) — Q2e, Q2f, Q2g"),
        spacer(0.08),
        q_header("Q2e — Impact of Similarity Metric on User-CF & Item-CF  (10 pts)"),
        body("Three similarity metrics were tested at K=40: "
             "<b>Cosine</b> (angle between rating vectors), "
             "<b>MSD</b> (1 / (1 + mean squared difference)), and "
             "<b>Pearson</b> (correlation coefficient)."),
        spacer(0.04),
        styled_table(
            ["Model", "Cosine RMSE", "MSD RMSE", "Pearson RMSE"],
            [["User-CF", "0.9931", "0.9678", "0.9983"],
             ["Item-CF", "0.9951", "0.9346", "0.9891"]],
            col_widths=[1.75*inch, 1.75*inch, 1.75*inch, 1.75*inch],
        ),
        spacer(0.04),
        img(sim_plot, width=7.0*inch),
        caption("Figure 5 — Effect of similarity metric on RMSE and MAE for User-CF and Item-CF (K=40)."),
        spacer(0.04),
        body("<b>Analysis:</b>"),
        bullet("<b>MSD performs best</b> for both User-CF (0.9678) and Item-CF (0.9346). "
               "It directly measures the average squared difference in ratings over common "
               "items/users, making it more sensitive to rating magnitude than directional metrics."),
        bullet("<b>Cosine</b> is the second-best for User-CF (0.9931) but slightly worse "
               "for Item-CF compared to Pearson."),
        bullet("<b>Pearson</b> performs worst for User-CF (0.9983) because mean-centering "
               "can overfit with sparse user–item overlap."),
        bullet("<b>Is the impact consistent?</b> Yes — MSD ranks first for both User-CF "
               "and Item-CF. The relative ordering (MSD > Cosine ≈ Pearson) is consistent, "
               "though the magnitude of improvement is larger for Item-CF."),
    ]

    # ── 2f ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q2f — Impact of Number of Neighbours K  (10 pts)"),
        body("K was varied from 5 to 80 using cosine similarity. "
             "5-fold cross-validation was applied at each K value."),
        spacer(0.04),
        styled_table(
            ["K", "User-CF RMSE", "Item-CF RMSE"],
            [["5",  "1.0440", "1.1009"],
             ["10", "1.0091", "1.0508"],
             ["20", "0.9966", "1.0171"],
             ["40", "0.9931", "0.9951"],
             ["80", "0.9937", "0.9808"]],
            col_widths=[2.3*inch, 2.3*inch, 2.4*inch],
        ),
        spacer(0.04),
        img(k_plot, width=7.0*inch),
        caption("Figure 6 — Effect of K on RMSE and MAE for User-CF and Item-CF (cosine similarity)."),
        spacer(0.04),
        body("<b>Analysis:</b>"),
        bullet("Both models improve significantly as K increases from 5 → 20. Very small K "
               "(e.g., 5) leads to high variance — predictions rely on too few neighbours."),
        bullet("Beyond K ≈ 40, User-CF performance plateaus — adding more neighbours "
               "introduces dissimilar users that add noise."),
        bullet("Item-CF continues improving up to K = 80, suggesting item similarities "
               "are more stable across users."),
    ]

    # ── 2g ────────────────────────────────────────────────────────────────────
    story += [
        q_header("Q2g — Best K for User-CF vs Item-CF  (10 pts)"),
        spacer(0.04),
        styled_table(
            ["Model", "Best K", "Best RMSE", "Observation"],
            [["User-CF", "K = 40", "0.9931", "Plateaus after K=40"],
             ["Item-CF", "K = 80", "0.9808", "Still improving at K=80"]],
            col_widths=[1.6*inch, 1.4*inch, 1.4*inch, 2.6*inch],
        ),
        spacer(0.04),
        body("<b>Conclusion:</b> The best K values are <b>not the same</b>. User-CF reaches "
             "its optimum at K = 40, while Item-CF benefits from a larger neighbourhood "
             "of K = 80. This is because item rating profiles are denser and more consistent "
             "across users — popular items receive many ratings, providing reliable similarity "
             "signals even at large K. User profiles, by contrast, are sparser and noisier, "
             "so a smaller K prevents averaging over dissimilar users."),
        spacer(0.06),
        hr(),
    ]

    # ── Code link ─────────────────────────────────────────────────────────────
    story += [
        spacer(0.06),
        body("<b>Code Repository:</b> All code for this assignment is available at:"),
        Paragraph(
            "<font color='#1a4fa0'><u>https://github.com/Brijesh03032001/HW3_CSE572_BrijeshKumar</u></font>",
            ParagraphStyle("link", fontSize=10, fontName="Helvetica",
                           textColor=BLUE, spaceBefore=4, spaceAfter=4)
        ),
        note("Scripts: kmeans_scratch.py (Task 1) and Recommendation_system_fast.py (Task 2). "
             "Virtual environment: homew3 (Python 3.11). "
             "Key libraries: numpy, pandas, scikit-learn, scikit-surprise, matplotlib."),
        spacer(0.1),
    ]

    # ── Footer banner ─────────────────────────────────────────────────────────
    footer_data = [[Paragraph(
        "Brijesh Kumar  •  CSE 572 Data Mining  •  HW3  •  Arizona State University  •  Spring 2026",
        FOOTER
    )]]
    footer_tbl = Table(footer_data, colWidths=[7.0*inch])
    footer_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), LBLUE),
        ("TOPPADDING",   (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
    ]))
    story += [footer_tbl]

    doc.build(story)
    print(f"\n✅  PDF saved → {PDF_OUT}")


if __name__ == "__main__":
    build_pdf()
