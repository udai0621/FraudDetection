"""
src/evaluate.py 

モデル評価スクリプト

実務ポイント:
- Accuracy は使わない (不均衡データでは無意味)
- AUC-ROC と Precision-Recall 曲線をセットで確認する。
- 閾値 (threshold) の選び方をビジネス視点で考える
  -> 「見逃しコスト」vs「誤検知コスト」のトレードオフ
"""

#=================================================================
# Library
#=================================================================

# Standard
import joblib
from pathlib import Path
import sys
from typing import Tuple
sys.path.append(str(Path(__file__).parent.parent))

# 3rd party
import japanize_matplotlib
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")   # サーバー環境での描画
import seaborn as sns 
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score
)


#=================================================================
# 定数
#=================================================================

OUTPUT_DIR = "outputs"


#=================================================================
# Codes
#=================================================================

def load_artifacts(output_dir: str = OUTPUT_DIR):
    """学習済みモデルとテストデータの読込"""
    model = joblib.load(f"{output_dir}/model_xgboost.pkl")
    X_test = np.load(f"{output_dir}/X_test.npy")
    y_test = np.load(f"{output_dir}/y_test.npy")
    return model, X_test, y_test


def plot_roc_curve(
    y_test: np.ndarray, 
    y_prob: np.ndarray, 
    output_dir: str = OUTPUT_DIR
) -> float:
    """
    ROC 曲線の描画

    AUC-ROC の読み方:
    - 1.0 = 完璧, 0.5 = ランダム (コイントス)
    - 実務では 0.90 以上を目指す
    - ただし不均衡が激しい場合は PR 曲線も必ず確認する
    """
    fpr, tpr, threshold = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")

    ax.set_xlabel("False Positive Rate\n(正常取引を不正と誤検知する割合)", fontdict=dict(size=11))
    ax.set_ylabel("True Positive (Recall)\n(実際の不正を検知できる割合)", fontdict=dict(size=11))
    ax.set_title("ROC 曲線", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{output_dir}/roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📈 ROC曲線保存: {path} (AUC = {auc:.4f})")
    
    return auc


def plot_pr_curve(
    y_test: np.ndarray, 
    y_prob: np.ndarray, 
    output_dir: str = OUTPUT_DIR
) -> float:
    """
    Precision-Recall 曲線の描画

    なぜ不均衡データに重要か:
    ROC 曲線は負例(正常取引)の多さに引きずられて楽観的になりがち。
    PR 曲線は正例(不正取引)にフォーカスするため、実際の検知性能を正確に反映する。

    AP (Average Precision) の読み方:
    - 1.0 = 完璧、ランダムなら不正率(≒ 0.005)に近い値になる
    """
    precision, recall, threshold = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"XGBoost (AP = {ap:.4f})")
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        lw=1,
        label=f"Baseline (不正率 {baseline:.3%})"
    )
    ax.fill_between(recall, precision, alpha=0.1, color="darkorange")

    ax.set_xlabel("Recall (実際の不正を何割検知できるか)", fontsize=11)
    ax.set_ylabel("Precision (不正と判定したうち本当に不正の割合)", fontsize=11)
    ax.set_title("Precision-Recall 曲線", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{output_dir}/pr_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📈 PR曲線保存: {path} (AP = {ap:.4f})")
    
    return ap


def plot_confusion_matrix(
    y_test: np.ndarray, 
    y_pred: np.ndarray, 
    threshold: float,
    output_dir: str = OUTPUT_DIR
) -> None:
    """混同行列の描画"""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["正常(予測)", "不正(予測)"],
        yticklabels=["正常(実際)", "不正(実際)"],
        annot_kws=dict(size=14)
    )
    ax.set_title(f"混同行列 (閾値: {threshold:.2f})", fontsize=13, fontweight="bold")
    ax.set_ylabel("実際のラベル", fontsize=11)
    ax.set_xlabel("予測のラベル", fontsize=11)

    # TP/FP/TN/FN の説明をコメントで追加
    textstr = (
        f"TN={tn:,} FP={fp:,}\n"
        f"FN={fn:,} TP={tp:,}\n\n"
        f"FP = 正常をブロック(顧客クレーム)\n"
        f"FN = 不正見逃し(損失)"
    )
    ax.text(
        x=0.35, y=0.5, s=textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 混同行列保存: {path}")


def find_best_threshold(
    y_test: np.ndarray, 
    y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    F1スコアが最大になる閾値を探索する。

    実務では F1 最大とは限らない。
    「不正見逃し1件 = 顧客クレーム何件分」という
    ビジネスコストに応じて調整する。
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [
        f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_thresh, best_f1


def evaluate(output_dir: str = OUTPUT_DIR):
    """評価の全工程"""
    print("\n" + "=" * 60)
    print("モデル評価")
    print("=" * 60)

    model, X_test, y_test = load_artifacts(output_dir)
    y_prob = model.predict_proba(X_test)[:, 1]  # 不正確率スコア

    # ROC・PR 曲線 
    auc = plot_roc_curve(y_test, y_prob, output_dir)
    ap = plot_pr_curve(y_test, y_prob, output_dir)

    # 最適閾値の探索
    best_thresh, best_f1 = find_best_threshold(y_test, y_prob)
    print(f"\n🎯 最適閾値: {best_thresh:.2f} (F1={best_f1:.4f})")

    y_pred = (y_prob >= best_thresh).astype(int)
    plot_confusion_matrix(y_test, y_pred, best_thresh, output_dir)

    # 分類レポート
    print("\n📝 Classification Report")
    print(classification_report(
        y_test, y_pred, target_names=["正常", "不正"], digits=4
    ))

    print("\n" + "=" * 60)
    print("✅ 評価完了")
    print(f"   AUC-ROC : {auc:.4f}")
    print(f"   AP(PR)  : {ap:.4f}")
    print(f"   Best F1 : {best_f1:.4f} at threshold={best_thresh:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
