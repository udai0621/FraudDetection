"""
src/explain.py 

SHAP (SHapley Additive exPlanations) による説明可能AI モジュール。

なぜ SHAP が実務で重要か:
1. 規制対応: 金融庁・バーゼル規制で「モデルの説明責任」が求められる。
2. 顧客対応: 「なぜあなたの取引は止められたか」を説明できる。
3. デバッグ: モデルが間違ったものを重視していないか確認できる。
4. 特徴量選択: 不要な特徴量の特定と削除

SHAP 値の解釈:
- プラス値 -> 不正確率を上げる方向に働いた特徴量
- マイナス値 -> 不正確率を上げる方向に働いた特徴量
- 絶対値が大きいほど、その特徴量の影響が強い
"""

#=================================================================
# Library
#=================================================================

# Standard
from pathlib import Path
import sys
from typing import Tuple, Any

# 3rd party
import japanize_matplotlib
import joblib
import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
import shap


#=================================================================
# Global Define
#=================================================================

sys.path.append(str(Path(__file__).parent.parent))
matplotlib.use("Agg")
OUTPUT_DIR = "outputs"


#=================================================================
# Codes
#=================================================================

def load_artifacts(
    model_file: str, 
    output_dir: str = OUTPUT_DIR
) -> Tuple[Any, np.ndarray, np.ndarray, list]:
    """
    以下各データの読込
    - 学習済みモデル
    - テストデータ
    - 特徴量のリスト

    Parameters
    """
    model = joblib.load(f"{output_dir}/{model_file}")
    X_test = np.load(f"{output_dir}/X_test.npy")
    y_test = np.load(f"{output_dir}/y_test.npy")
    feature_cols = joblib.load(f"{output_dir}/feature_cols.pkl")
    return model, X_test, y_test, feature_cols


def compute_shap_values(
    model: Any,
    X_test: np.ndarray,
    sample_size: int = 2000
) -> Tuple[shape.TreeExplainer, np.ndarray, np.ndarray, np.ndarray]:
    """
    SHAP 値の計算。

    TreeExplainer: XGBoost / LightGBM / RandomForest 等 
    Tree系モデルに特化した高速計算メソッド

    sample_size: 全テストデータだと時間がかかるためサンプリング
    """
    print(f"🔍 SHAP値計算中 (サンプル数: {sample_size}件) ...")
    idx = np.random.choice(
        len(X_test), 
        size=min(sample_size, len(X_test)),
        replace=False
    )
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print("✅ SHAP値計算完了")
    return explainer, shap_values, X_sample, idx


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_cols: list,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    SHAP Summary Plot (Beeswarm)

    読み方:
    - 縦軸: 特徴量 (上ほど影響が大きい)
    - 横軸: SHAP値 (右=不正方向、左=正常方向)
    - 色: 特徴量の値 (赤=高い、青=低い)

    例: amount が赤(高い) + 右(不正方向) なら 
        「金額が高いほど不正確率が上がる」と解釈できる
    """
    plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_cols,
        show=False,
        plot_type="dot",
        max_display=12
    )
    plt.title(
        "SHAP Summary Plot\n (縦: 特徴量の重要度順 / 横: 不正確率への影響)",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    path = f"{output_dir}/shape_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 SHAP Summary保存: {path}")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_cols: list,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    特徴量重要度の棒グラフ (SHAP mean |value|)
    プレゼン・報告書向けに分かりやすい形式。
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = sorted(zip(feature_cols, mean_abs_shap),
                           key=lambda x: x[1], reverse=True)
    features, importances = zip(*importance_df)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        range(len(features)),
        importances,
        color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(features)))
    )
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP値| (不正確率への平均影響度)", fontsize=11)
    ax.set_title("特徴量重要度 (SHAP ベース)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # 値をバーの横に表示
    for i, v in enumerate(importances):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = f"{output_dir}/shap_importance_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 SHAP重要度棒グラフ保存: {path}")


def plot_shap_waterfall(
    explainer: shap.TreeExplainer | shap.Explainer,
    shap_values: np.ndarray, 
    X_sample: np.ndarray, 
    y_test_sample: np.ndarray,
    feature_cols: list,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Waterfall Plot: 1件の取引について「なぜ不正と判定したか」を説明。

    実務での使用場面:
    - 顧客から「なぜ取引が止められたのか」と問い合わせがあった時
    - 審査担当者が個別ケースをレビューするとき
    - コンプライアンス・内部監査での説明資料

    実際の不正取引サンプルを1件選んで説明する。
    """
    # 実際の不正取引のインデックスを取得
    fraud_indices = np.where(y_test_sample == 1)[0]
    if len(fraud_indices) == 0:
        print("⚠️ サンプル内に不正取引がなかったためスキップ")
        return

    sample_idx = fraud_indices[0]   

    # Explanation オブジェクトを作成
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_sample[sample_idx],
        feature_names=feature_cols
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title(
        f"Waterfall Plot (不正取引 サンプル #{sample_idx} の判定理由)",
        fontsize=12, fontweight="bold", pad=10
    )
    plt.tight_layout()
    path = f"{output_dir}/shap_waterfall_sample.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Waterfall Plot保存: {path}")
    print(f"    -> 取引 #{sample_idx} の各特徴量が不正スコアを上下させた量がわかります。")


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_cols: list,
    target_feature: str = "amount",
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Dependence Plot: 特定の特徴量と SHAP 値の関係を可視化。

    「取引金額が増えるにつれ、不正確率がどう変化するか」を確認できる。
    モデルが学習した「ルール」を視覚的に理解するのに役立つ。
    """
    if target_feature not in feature_cols:
        print(f"⚠️ {target_feature} が特徴量に存在しません。")
        return

    feat_idx = list(feature_cols).index(target_feature)
    print(feature_cols, target_feature, feat_idx)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx, shap_values, X_sample,
        feature_names=feature_cols,
        ax=ax, show=False
    )
    ax.set_title(
        f"Dependence Plot: {target_feature}\n(この特徴量の値と不正確率への影響の関係)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{output_dir}/shap_dependence_{target_feature}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Dependence Plot保存: {path}")


def explain(output_dir: str = OUTPUT_DIR):
    """SHAP の全工程"""
    print("\n" + "=" * 60)
    print("SHAP による説明可能AI")
    print("=" * 60)

    model, X_test, y_test, feature_cols = load_artifacts(model_file="model_xgboost.pkl")
    explainer, shap_values, X_sample, idx = compute_shap_values(model, X_test)
    y_test_sample = y_test[idx]

    # 各プロットの生成
    plot_shap_summary(shap_values, X_sample, feature_cols, output_dir)
    plot_shap_bar(shap_values, feature_cols, output_dir)
    plot_shap_waterfall(
        explainer, 
        shap_values,
        X_sample,
        y_test_sample,
        feature_cols,
        output_dir
    )
    plot_shap_dependence(
        shap_values,
        X_sample,
        feature_cols,
        target_feature="amount",
        output_dir=output_dir
    )
    plot_shap_dependence(
        shap_values,
        X_sample,
        feature_cols,
        target_feature="velocity_1h",
        output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("✅ SHAP 分析完了")
    print("   outputs/ に以下のファイルが生成されました:")
    print("   - shap_summary.png          : 全体の特徴量重要度")
    print("   - shap_importance_bar.png   : 棒グラフ形式の重要度")
    print("   - shap_waterfall_sample.png : 1件の判定理由")
    print("   - shap_dependence_amount.png: 金額と不正確率の関係")
    print("   - shap_dependence_velocity_1h.png: 速度と不正確率の関係")
    print("=" * 60)


if __name__ == "__main__":
    explain()


