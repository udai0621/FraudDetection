"""
src/preprocess.py 

前処理・特徴量エンジニアリングモジュール。

実務ポイント:
- SMOTE はここでは適用しない(train.py で train データのみに適用)
- テストデータへの情報漏洩 (Data Leak) を防ぐため、
  StandardScaler の fit も必ず訓練データのみで行う
"""

#=================================================================
# Library
#=================================================================

# Standard
import joblib
from pathlib import Path

# 3rd party
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#=================================================================
# 定数
#=================================================================

FEATURE_COLS = [
    "amount",
    "hour",
    "days_since_last_txn",
    "merchant_risk_score",
    "is_foreign",
    "velocity_1h",
    "velocity_24h",
    "amt_vs_avg_ratio"
]
TARGET_COL = "is_fraud"


#=================================================================
# Codes
#=================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """CSVを読み込み、基本的な型チェックを行う"""
    df = pd.read_csv(filepath)
    assert TARGET_COL in df.columns, f"{TARGET_COL} 列が見つかりません。"
    print(f"✅ データ読込完了: {len(df):,} 件")
    print(f"    不正率: {df[TARGET_COL].mean():.3%}") 
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特徴量エンジニアリング。
    実務では「ドメイン知識」でここを充実されることが最重要。
    """
    df = df.copy() 

    # 深夜フラグ (0〜5時 or 22〜23時)
    df["is_night"] = df["hour"].apply(lambda h: 1 if h <= 5 or h >= 22 else 0)

    # 高額フラグ
    threshold_95 = df["amount"].quantile(0.95)
    df["is_high_amount"] = (df["amount"] > threshold_95).astype(int)

    # 速度 x 金額の交互作用特徴量 (短時間に高頻度 + 高額 -> 危険)
    df["velocity_amount_interaction"] = df["velocity_1h"] * df["amount"]

    # リスクスコア x 海外フラグ
    df["foreign_risk"] = df["is_foreign"] * df["merchant_risk_score"]

    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    学習・テスト分割 + 標準化。

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # 特徴量エンジニアリング後の列リスト
    feature_cols = FEATURE_COLS + [
        "is_night", "is_high_amount",
        "velocity_amount_interaction", "foreign_risk"
    ]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # 分割（層化サンプリング: 不正率を train/test で同じに保つ）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,            # ← 不均衡データでは必須
        random_state=random_state
    )

    # スケーリング: fit は必ず train のみ
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   # transform のみ（leak 防止）

    print(f"\n📊 データ分割結果")
    print(f"   Train: {len(X_train):,}件  不正率: {y_train.mean():.3%}")
    print(f"   Test : {len(X_test):,}件   不正率: {y_test.mean():.3%}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_scaler(scaler: StandardScaler, output_dir: str = "outputs"):
    """学習済みスケーラーを保存（本番推論で使い回すため）"""
    Path(output_dir).mkdir(exist_ok=True)
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f"💾 スケーラー保存: {output_dir}/scaler.pkl")

