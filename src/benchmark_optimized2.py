#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark.py  -  Evaluation and visualization for the HGT HMM project.

Example (as in README):

    python src/benchmark.py \
        --predictions results/predictions.tsv \
        --output results/benchmark \
        --fasta data/genome.fasta

功能：
    1. 读入预测结果 TSV（GeneID, State, Prob_Host, Prob_Ameliorated, Prob_Foreign）
    2. （如果有真值）计算三分类混淆矩阵 + Host vs Non-Host ROC / AUC
    3. 画出：
        - confusion_matrix.png
        - roc_curve.png
        - heatmap.png（后验概率热图）
        - barplot.png（Top 外源候选基因）
"""

import argparse
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn 用于混淆矩阵 & ROC，如果没有安装，会自动跳过对应图
try:
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

# Biopython 用于按 FASTA 顺序排列基因，缺了也没关系
try:
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover
    BIOPYTHON_AVAILABLE = False


STATE_ORDER = [0, 1, 2]
STATE_LABELS = {0: "Host", 1: "Ameliorated", 2: "Foreign"}


# ---------- 小工具函数 ----------

def ensure_dir(path: str) -> None:
    """创建输出目录（若不存在）。"""
    os.makedirs(path, exist_ok=True)


def load_predictions(path: str) -> pd.DataFrame:
    """读取预测文件，并检查必须字段是否存在。"""
    df = pd.read_csv(path, sep="\t")

    required = ["GeneID", "State", "Prob_Host", "Prob_Ameliorated", "Prob_Foreign"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Predictions file '{path}' is missing required columns: {missing}"
        )

    # 确保状态是整数
    try:
        df["State"] = df["State"].astype(int)
    except Exception:
        # 如果不是数字，可以在这里加映射（例如 'Host' -> 0），当前假设已经是 0/1/2
        pass

    return df


def attach_truth(df: pd.DataFrame, truth_path: Optional[str] = None) -> pd.DataFrame:
    """
    将真实标签附着到 df 上，统一叫 'TrueState'。

    优先级：
      1. 如果提供了 truth_path，就从该文件合并到 df 中；
      2. 否则在 df 本身找可能的真值列（如 TrueState / state_true / true_label）。
    """
    df = df.copy()

    if truth_path is not None:
        truth = pd.read_csv(truth_path, sep="\t")
        if "GeneID" not in truth.columns:
            raise ValueError("Truth file must contain a 'GeneID' column.")

        # 尝试找一个合理的真值列名
        candidates = []
        for col in truth.columns:
            name = col.lower()
            if name in ("truestate", "true_state", "label", "true_label", "state"):
                candidates.append(col)

        if not candidates:
            raise ValueError(
                "Truth file must contain a column with true labels "
                "(e.g. 'TrueState' or 'state')."
            )

        col = candidates[0]
        truth = truth[["GeneID", col]].rename(columns={col: "TrueState"})
        merged = df.merge(truth, on="GeneID", how="left")

        # 尽量转成整数
        try:
            merged["TrueState"] = merged["TrueState"].astype(int)
        except Exception:
            pass
        return merged

    # 没有传 truth_path，就在 df 里找
    candidates = []
    for col in df.columns:
        name = col.lower()
        if name in ("truestate", "true_state", "state_true", "true_label", "label_true"):
            candidates.append(col)

    if candidates:
        df = df.rename(columns={candidates[0]: "TrueState"})
        try:
            df["TrueState"] = df["TrueState"].astype(int)
        except Exception:
            pass

    return df


def gene_order_from_fasta(fasta_path: str) -> Sequence[str]:
    """
    从 FASTA 文件中读取记录顺序（record.id），
    用于按基因组位置排序基因。
    """
    if not BIOPYTHON_AVAILABLE:
        print("Biopython not installed; ignoring --fasta for ordering.")
        return []

    ids = []
    with open(fasta_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
    return ids


def order_dataframe(df: pd.DataFrame, fasta_path: Optional[str]) -> pd.DataFrame:
    """
    若提供 FASTA 文件，则按 FASTA 中记录顺序排序 GeneID；
    否则保持原有顺序。
    """
    df = df.copy()

    if fasta_path is None or not os.path.exists(fasta_path):
        return df.reset_index(drop=True)

    fasta_ids = gene_order_from_fasta(fasta_path)
    if not fasta_ids:
        return df.reset_index(drop=True)

    # 使用 Categorical 来指定排序顺序，未出现在 FASTA 中的排在最后
    cat = pd.Categorical(df["GeneID"], categories=fasta_ids, ordered=True)
    df["_order"] = cat
    df = df.sort_values("_order").drop(columns=["_order"])
    return df.reset_index(drop=True)


# ---------- 画图函数 ----------

def plot_confusion(df: pd.DataFrame, out_path: str) -> None:
    """画 3×3 混淆矩阵，并顺便输出一个 text 报告。"""
    if "TrueState" not in df.columns:
        print("No 'TrueState' column found; skipping confusion matrix.")
        return
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not installed; skipping confusion matrix.")
        return

    y_true = df["TrueState"].values
    y_pred = df["State"].values

    cm = confusion_matrix(y_true, y_pred, labels=STATE_ORDER)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_xticks(range(len(STATE_ORDER)))
    ax.set_yticks(range(len(STATE_ORDER)))
    ax.set_xticklabels([STATE_LABELS[s] for s in STATE_ORDER])
    ax.set_yticklabels([STATE_LABELS[s] for s in STATE_ORDER])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    # 在格子里标数
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # 文本报告（precision/recall/F1）
    report = classification_report(
        y_true,
        y_pred,
        labels=STATE_ORDER,
        target_names=[STATE_LABELS[s] for s in STATE_ORDER],
    )
    report_path = os.path.splitext(out_path)[0] + "_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved confusion matrix to {out_path}")
    print(f"Saved classification report to {report_path}")


def plot_roc_host_vs_nonhost(df: pd.DataFrame, out_path: str) -> None:
    """
    Host vs Non-Host (Ameliorated + Foreign) 的 ROC 曲线。
    使用 Prob_Host 作为打分。
    """
    if "TrueState" not in df.columns:
        print("No 'TrueState' column found; skipping ROC.")
        return
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not installed; skipping ROC.")
        return

    # Host = 1, Non-Host = 0
    y_true = (df["TrueState"].values == 0).astype(int)
    scores = df["Prob_Host"].values

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")  # 随机分类基线
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Host vs Non-Host ROC")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved ROC curve to {out_path}")


def plot_posterior_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """
    画 posterior probability 的热图：
        y 轴：3 个状态
        x 轴：基因索引（已经按基因组排序）
    """
    probs = df[["Prob_Host", "Prob_Ameliorated", "Prob_Foreign"]].T.values

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(probs, aspect="auto", interpolation="nearest")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Host", "Ameliorated", "Foreign"])
    ax.set_xlabel("Gene index (along genome)")
    ax.set_title("Posterior state probabilities")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Posterior probability")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved posterior heatmap to {out_path}")


def plot_top_foreign_barplot(df: pd.DataFrame, out_path: str, top_k: int = 20) -> None:
    """
    画 Top-K 外源候选基因柱状图（按 Prob_Foreign 排序）。
    """
    df_sorted = df.sort_values("Prob_Foreign", ascending=False).head(top_k)

    if df_sorted.empty:
        print("No data to plot barplot (empty predictions).")
        return

    fig_width = max(6, len(df_sorted) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    x = np.arange(len(df_sorted))
    ax.bar(x, df_sorted["Prob_Foreign"].values)

    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["GeneID"].astype(str).values, rotation=90)
    ax.set_ylabel("P(Foreign)")
    ax.set_title(f"Top {len(df_sorted)} foreign gene candidates")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved barplot to {out_path}")


# ---------- 主入口 ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation and visualization for HGT HMM predictions."
    )
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="TSV file with columns: GeneID, State, Prob_Host, Prob_Ameliorated, Prob_Foreign.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for benchmark plots.",
    )
    parser.add_argument(
        "--fasta",
        "-f",
        default=None,
        help="Genome FASTA file (used only to order genes along the genome).",
    )
    parser.add_argument(
        "--truth",
        "-t",
        default=None,
        help=(
            "Optional TSV file with true labels. "
            "Must contain 'GeneID' and a label column (e.g. 'TrueState')."
        ),
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=20,
        help="Number of top foreign candidates shown in the barplot (default: 20).",
    )

    args = parser.parse_args()

    ensure_dir(args.output)

    # 1. 读预测结果
    df = load_predictions(args.predictions)

    # 2. 附加真值（如果路径或列存在）
    df = attach_truth(df, args.truth)

    # 3. 按基因组顺序排序（如果给了 FASTA）
    df = order_dataframe(df, args.fasta)

    # 4. 画各种图
    plot_confusion(df, os.path.join(args.output, "confusion_matrix.png"))
    plot_roc_host_vs_nonhost(df, os.path.join(args.output, "roc_curve.png"))
    plot_posterior_heatmap(df, os.path.join(args.output, "heatmap.png"))
    plot_top_foreign_barplot(
        df,
        os.path.join(args.output, "barplot.png"),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
