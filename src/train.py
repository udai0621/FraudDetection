"""
src/train.py 

モデル学習スクリプト

実務ポイント:
1. SMOTE は「訓練データのみ」に適用 (テストデータへの leak を防ぐ)
2. XGBoost の scale_pos_weight で不均衡を一次補正した後、SMOTE で強化
3. 閾値 (threshold) はビジネスコストに応じて後から調整する
"""

#=================================================================
# Library
#=================================================================

# Standard
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# 3rd party
from imblearn.over_sampling import SMOTE
import numpy as np 
from xgboost import XGBClassifier

# Custom
from src.preprocess import load_data, add_features, split_and_scale, save_scaler


#=================================================================
# 定数
#=================================================================

DATA_PATH = "data/transactions.csv"
OUTPUT_DIR = "outputs"


#=================================================================
# Codes
#=================================================================

def build_model(scale_pos_weight: float = 10.0) -> XGBClassifier:
    """
    XGBoost モデルの定義

    重要なハイパーパラメータ:
    - scale_pos_weight: 不正クラスの重みを上げる(不均衡補正の1つ目の手段)
    - max_depth       : 木の深さ。深すぎると過学習。
    - learning_rate   : 学習率。小さいほど安定するが時間がかかる。
    - n_estimators    : 木の本数。early_stopping と組み合わせて決める。
    - eval_metric     : 'aucpr' = Precision-Recall AUC (不均衡データ向き)
    """
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # 不正クラスを重視
        eval_metric="aucpr",                # PR-AUCで学習を監視
        early_stopping_rounds=30,           # 改善がなければ早期終了
        random_state=42,
        n_jobs=-1,
    )

    return model


def train(data_path: str = DATA_PATH, output_dir: str = OUTPUT_DIR):
    """学習の全工程"""
    Path(output_dir).mkdir(exist_ok=True)

    # 1. データ読込・特徴量エンジニアリング
    print("\n" + "=" * 60)
    print("STEP 1: データ読み込み・特徴量エンジニアリング")
    print("=" * 60)
    df = load_data(data_path)
    df = add_features(df)

    # 2. 分割・スケーリング
    print("\n" + "=" * 60)
    print("STEP 2: 学習/テスト分割 + 標準化")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, feature_cols = split_and_scale(df)
    save_scaler(scaler, output_dir)

    # 3. SMOTE で不均衡データを補正 (訓練データのみ)
    print("\n" + "=" * 60)
    print("STEP 3: SMOTE による不均衡補正 (訓練データのみ)")
    print("=" * 60)
    print(f"    SMOTE 前: 正常={sum(y_train==0):,} 不正={sum(y_train==1):,}")

    smote = SMOTE(
        sampling_strategy=0.1,  # 不正:正常 = 1:10 になるように補完
        random_state=42,
        k_neighbors=5
    )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"    SMOTE 後: 正常={sum(y_train_res==0):,} 不正={sum(y_train_res==1):,}")
    print(f"    ※ テストデータは元のまま (leak 防止)")

    # 4. scale_pos_weight の計算
    #    不正が増えた後の比率で再計算
    neg = sum(y_train_res == 0)
    pos = sum(y_train_res == 1)
    spw = neg / pos
    print(f"\n  scale_pos_weight = {spw:.2f}")

    # 5. モデル学習
    print("\n" + "=" * 60)
    print("STEP 4: XGBoost 学習開始")
    print("=" * 60)
    model = build_model(scale_pos_weight=spw)

    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=50  # 50ラウンドごとに進捗表示
    )

    best_iter = model.best_iteration
    print(f"\n✅ 学習完了 (最適ラウンド: {best_iter})")

    # 6. 保存
    model_path = f"{output_dir}/model_xgboost.pkl"
    joblib.dump(model, model_path)
    print(f"💾 モデル保存: {model_path}")

    # テストデータも保存 (evaluate.py / explain.py で使い回す)
    np.save(f"{output_dir}/X_test.npy", X_test)
    np.save(f"{output_dir}/y_test.npy", y_test)
    joblib.dump(feature_cols, f"{output_dir}/feature_cols.pkl")
    print(f"💾 テストデータ保存完了")

    return model, X_test, y_test, feature_cols


if __name__ == "__main__":
    train()
