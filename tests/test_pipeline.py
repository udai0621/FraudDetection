"""
tests/test_pipeline.py 

パイプライン全体の動作確認テスト。
CI/CD に組み込んでモデル品質を継続的に監視することを想定。

実務ポイント:
- AUC-ROC が一定水準を下回ったら自動でアラートを出す
- データ生成の統計的整合性(不正率など)を確認する
- 特徴量の件数が変わっていないか(スキーマチェック)を確認する
"""

#=================================================================
# Library
#=================================================================

# Standard
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 3rd party
from matplotlib.pyplot import cla
import numpy as np 
import pandas as pd 
import pytest 
from sklearn.metrics import roc_auc_score


#=================================================================
# Test codes: データ生成
#=================================================================

class TestDataGeneration:
    """データ生成スクリプトの動作確認"""

    def setup_method(self):
        """各テスト前にデータを生成"""
        from data.generate_data import generate_fraud_dataset
        self.df = generate_fraud_dataset(n_samples=5000, fraud_rate=0.005)

    def test_columns_exist(self):
        """必要な列が全て存在するか"""
        required_cols = [
            "transaction_id", "amount", "hour", "days_since_last_txn",
            "merchant_risk_score", "is_foreign", "velocity_1h",
            "velocity_24h", "amt_vs_avg_ratio", "is_fraud"
        ]
        for col in required_cols:
            assert col in self.df.columns, f"列 '{col}' が存在しません"

    def test_fraud_rate(self):
        """不正率が期待範囲内(0.1%〜2%)か"""
        fraud_rate = self.df["is_fraud"].mean()
        assert 0.001 <= fraud_rate <= 0.02, \
            f"不正率が想定外: {fraud_rate:.3%}"

    def test_no_null_values(self):
        """欠損値がないか"""
        assert self.df.isnull().sum().sum() == 0, "欠損値が存在します"

    def test_hour_range(self):
        """時刻が 0〜23 の範囲か"""
        assert self.df["hour"].between(0, 23).all(), \
            "time列に0〜23の範囲外の値があります"

    def test_amount_positive(self):
        """取引金額が正の値か"""
        assert (self.df["amount"] > 0).all(), "取引金額に0以下の値があります。"

    def test_binary_flags(self):
        """2値グラフ列が 0 or 1 のみか"""
        for col in ["is_foreign", "is_fraud"]:
            unique_vals = set(self.df[col].unique())
            assert unique_vals.issubset({0, 1}), \
                f"'{col}' に0/1以外の値が含まれています: {unique_vals}"



#=================================================================
# Test codes: 前処理
#=================================================================

class TestPreprocessing:
    """前処理モジュールの動作確認"""

    def setup_method(self):
        from data.generate_data import generate_fraud_dataset
        from src.preprocess import add_features
        df_raw = generate_fraud_dataset(n_samples=5000)
        self.df = add_features(df_raw)

    def test_engineered_features_exist(self):
        """特徴量エンジニアリングで追加された列が存在するか"""
        new_cols = ["is_night", "is_high_amount",
                    "velocity_amount_interaction", "foreign_risk"]
        for col in new_cols:
            assert col in self.df.columns, f"特徴量 '{col}' が生成されていません"

    def test_is_night_binary(self):
        """is_night が 0/1 のみか"""
        assert set(self.df["is_night"].unique()).issubset({0, 1})

    def test_train_test_split_stratified(self):
        """層化サンプリング: 不正率が train/test で近いか"""
        from src.preprocess import split_and_scale
        X_train, X_test, y_train, y_test, _, _ = split_and_scale(self.df)

        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # 2倍以上乖離していなければOK
        assert abs(train_rate - test_rate) < train_rate, \
            f"不正率が大きく乖離: train={train_rate:.3%}, test={test_rate:.3%}"

    def test_no_data_leak(self):
        """テストデータのスケーラーは train の統計のみを使っているか確認"""
        from src.preprocess import split_and_scale
        X_train, X_test, y_train, y_test, scaler, _ = split_and_scale(self.df)

        # scaler は fit_transform した train データに基づく
        # train の平均が ≈ 0、std が ≈ 1 になっているはず
        train_means = X_train.mean(axis=0)
        train_stds  = X_train.std(axis=0)
        assert np.allclose(train_means, 0, atol=0.1), \
            "標準化後の train 平均が0から大きく外れています"
        assert np.allclose(train_stds, 1, atol=0.1), \
            "標準化後の train 標準偏差が1から大きく外れています"


#=================================================================
# Test codes: モデルの品質テスト
#=================================================================

class TestModelQuality:
    """
    学習済みモデルの品質ゲートテスト。
    CI/CD で「AUC が閾値を下回ったらデプロイしない」などに活用
    """

    AUC_THRESHOLD = 0.85    # 最低限のAUC水準（要件に応じて調整）
    AP_THRESHOLD  = 0.30    # Precision-Recall AUC 最低水準

    @pytest.fixture(autouse=True)
    def load_model(self):
        """学習済みモデルの読込（なければスキップ）"""
        import joblib
        model_path = Path("outputs/model_xgboost.pkl")
        if not model_path.exists():
            pytest.skip("学習済みモデルが存在しません（先に train.py を実行してください）")
        self.model = joblib.load(model_path)
        self.X_test = np.load("outputs/X_test.npy")
        self.y_test = np.load("outputs/y_test.npy")

    def test_auc_roc_above_threshold(self):
        """AUC-ROC が基準値以上か"""
        from sklearn.metrics import roc_auc_score
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_prob)
        assert auc >= self.AUC_THRESHOLD, \
            f"AUC-ROC が基準値を下回りました: {auc:.4f} < {self.AUC_THRESHOLD:.4f}"

    def test_average_precision_above_threshold(self):
        """
        Precision-Recall AUC が基準値以上か
        """
        from sklearn.metrics import average_precision_score
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        ap = average_precision_score(self.y_test, y_prob)
        assert ap >= self.AP_THRESHOLD, \
            f"Average Precision が基準値を下回りました: {ap:.4f} < {self.AP_THRESHOLD}"

    def test_model_outputs_probability(self):
        """predict_proba の出力が 0〜1 の範囲か"""
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        assert y_prob.min() >= 0.0 and y_prob.max() <= 1.0, \
            "確率スコアが 0〜1 の範囲外です"

    def test_model_detects_fraud(self):
        """不正取引を最低限検知できているか（Recall > 0.5）"""
        from sklearn.metrics import recall_score
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= 0.3).astype(int)    # ゆるめの閾値でRecall確認
        recall = recall_score(self.y_test, y_pred)
        assert recall >= 0.5, \
            f"Recall が低すぎます: {recall:.4f} (不正の大半を見逃しています)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
