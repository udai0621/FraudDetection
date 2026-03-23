"""
data/generate_data.py 

実務を想定した疑似クレジットカード取引データを生成します。
本番環境では Kaggle の Credit Card Fraud Dataset などに置き換えてください。

データの設計思想：
- 不正率を約0.5%に設定(実務に近い極端な不均衡)
- 不正取引には「実際にありがちな特徴」を意図的に埋め込む
  Ex.) 深夜の高額取引、短時間での連続利用、海外での利用 etc...
"""

#=================================================================
# Library
#=================================================================

# Standard
from pathlib import Path

# 3rd party
import numpy as np 
import pandas as pd 


#=================================================================
# 定数
#=================================================================

RANDOM_SEED = 42
N_SAMPLES = 100_000     # 総取引件数
FRAUD_RATE = 0.005      # 不正率0.5%


#=================================================================
# Codes
#=================================================================

def generate_fraud_dataset(
    n_samples: int = N_SAMPLES,
    fraud_rate: float = FRAUD_RATE,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    不正検知用の擬似取引データを生成する。

    Parameters
    ---------- 
    n_samples  : 総レコード数 
    fraud_rate : 不正ラベルの割合
    random_seed: 再現性のためのシード
    
    Returns
    ------- 
    pd.DataFrame : 特徴量 + ラベル(is_fraud)
    """
    rng = np.random.default_rng(random_seed)
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    #-----------------------------------------------
    # 正常取引の生成
    #-----------------------------------------------
    normal = {
        # 取引金額: 対数正規分布(少額が多い)
        "amount": rng.lognormal(mean=3.5, sigma=1.2, size=n_normal).clip(1, 5000),

        # 取引時刻(0〜23時):日中が多い
        "hour": rng.choice(
            range(24),
            size=n_normal, 
            p=_hour_distribution(peak_night=False)
         ),
         
        # 前回の取引からの経過日数: 数日〜1ヶ月程度
        "days_since_last_txn": rng.exponential(scale=5, size=n_normal).clip(0, 90),

        # 加盟店リスクスコア: 低リスクが大半
        "merchant_risk_score": rng.beta(a=2, b=8, size=n_normal),

        # 海外取引フラグ: 10%程度 
        "is_foreign": rng.binomial(1, p=0.10, size=n_normal),

        # 直近1時間の取引回数
        "velocity_1h": rng.poisson(lam=1.2, size=n_normal).clip(0, 10),

        # 直近24時間の取引回数
        "velocity_24h": rng.poisson(lam=4.0, size=n_normal).clip(0, 30),

        # 平均取引額との比率
        "amt_vs_avg_ratio": rng.lognormal(mean=0.0, sigma=0.4, size=n_normal).clip(0.1, 10)
    }
    
    #-----------------------------------------------
    # 不正取引の生成 (実務上の特徴を意図的に付与)
    #-----------------------------------------------
    fraud = {
        # 不正は高額取引が多い
        "amount": rng.lognormal(mean=5.5, sigma=1.5, size=n_fraud).clip(50, 20000),

        # 深夜・早朝に集中
        "hour": rng.choice(
            range(24),
            size=n_fraud, 
            p=_hour_distribution(peak_night=True)
         ),
         
        # 直近に不審な取引が多い
        "days_since_last_txn": rng.exponential(scale=0.5, size=n_fraud).clip(0, 10),

        # 高リスク加盟店
        "merchant_risk_score": rng.beta(a=5, b=2, size=n_fraud),

        # 海外取引が多い (60%)
        "is_foreign": rng.binomial(1, p=0.60, size=n_fraud),

        # 短時間に連続利用 (velocity_attack)
        "velocity_1h": rng.poisson(lam=5.0, size=n_fraud).clip(0, 20),
        "velocity_24h": rng.poisson(lam=12.0, size=n_fraud).clip(0, 50),

        # 普段と大きく異なる金額
        "amt_vs_avg_ratio": rng.lognormal(mean=1.5, sigma=0.8, size=n_fraud).clip(0.1, 30)
    }

    df_normal = pd.DataFrame(normal)
    df_normal["is_fraud"] = 0 

    df_fraud = pd.DataFrame(fraud)
    df_fraud["is_fraud"] = 1 

    df = (
        pd.concat([df_normal, df_fraud], ignore_index=True)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )

    # transaction_id を付与 
    df.insert(0, "transaction_id", [f"TXN{str(i).zfill(7)}" for i in df.index])
    return df


def _hour_distribution(peak_night: bool) -> list:
    """
    24時間の取引確率分布を返す。
    peak_night=True : 深夜・早朝に偏るように設計(不正)
    peak_night=False: 日中に偏るように設計(正常)
    """
    # 24時間の出現回数を設定
    if peak_night:
        # 0〜5・22〜23時に集中
        weights = [8, 8, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 8]
    else:
        # 9〜21時に集中
        weights = [1, 1, 1, 1, 1, 2, 3, 5, 7, 9, 10, 10,
                   9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    total = sum(weights)
    return [w / total for w in weights]


#=================================================================
# 実行
#=================================================================

if __name__ == "__main__":
    output_path = Path(__file__).parent / "transactions.csv"
    df = generate_fraud_dataset() 

    df.to_csv(output_path, index=False)

    print("=" * 50)
    print("✅ データ生成完了")
    print(f"   保存先   : {output_path}")
    print(f"   総件数   : {len(df):,}")
    print(f"   不正件数 : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean():.2%})")
    print("=" * 50)
    print(df.head())
