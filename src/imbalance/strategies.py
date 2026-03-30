"""
src/imbalance/strategies.py
 
各手法の具体実装（ConcreteStrategy群）
 
実装手法一覧:
  1. NoResamplingStrategy        ── ベースライン（何もしない）
  2. RandomOverSamplingStrategy  ── ランダムオーバーサンプリング
  3. RandomUnderSamplingStrategy ── ランダムアンダーサンプリング
  4. SmoteStrategy               ── SMOTE
  5. AdasynStrategy              ── ADASYN
  6. CombinedStrategy            ── Over + Under 組み合わせ
  7. ClassWeightStrategy         ── クラス重み付け
 
すべて sklearn のみで実装（外部ライブラリ不要）。
imbalanced-learn が使える環境では SmoteStrategy / AdasynStrategy を
ライブラリ版に差し替えることを推奨。
"""

#=================================================================
# Library
#=================================================================

# 3rd party 
from types import new_class
import numpy as np
from numpy.random import rand
from sklearn import neighbors
from sklearn.metrics.pairwise import kernel_metrics
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors

# Custom
from .base import ImbalanceStrategy, ResamplingResult
 
 
#=================================================================
# 1. ベースライン（何もしない）
#=================================================================
 
class NoResamplingStrategy(ImbalanceStrategy):
    """
    何も補正しない。
    他の手法と比較するときの「基準値」として使う。
    """
 
    @property
    def name(self) -> str:
        return "NoResampling（補正なし）"
 
    @property
    def category(self) -> str:
        return "baseline"
 
    def apply(self, X_train, y_train) -> ResamplingResult:
        return self._make_result(X_train, y_train, X_train, y_train)
 
 
#=================================================================
# 2. ランダムオーバーサンプリング
#=================================================================
 
class RandomOverSamplingStrategy(ImbalanceStrategy):
    """
    少数クラスのサンプルをランダムに「複製」して増やす。
 
    メリット  : シンプル・高速・情報損失なし
    デメリット: 同じデータの複製なので過学習しやすい
 
    例え話: 不正取引のレシートをコピー機で増やすイメージ。
            内容は全く同じなので多様性は生まれない。
    """
 
    def __init__(self, sampling_strategy: float = 0.1, random_state: int = 42):
        """
        Parameters
        ----------
        sampling_strategy : 補正後の 少数/多数 比率
                            0.1 → 少数:多数 = 1:10 になるよう複製
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
 
    @property
    def name(self) -> str:
        return "RandomOverSampling（ランダム複製）"
 
    @property
    def category(self) -> str:
        return "oversampling"
 
    def apply(self, X_train, y_train) -> ResamplingResult:
        minority_mask = y_train == 1
        majority_mask = y_train == 0
 
        X_min = X_train[minority_mask]
        y_min = y_train[minority_mask]
        X_maj = X_train[majority_mask]
        y_maj = y_train[majority_mask]
 
        # 目標の少数クラス件数
        n_majority = len(X_maj)
        n_target = int(n_majority * self.sampling_strategy)
        n_target = max(n_target, len(X_min))  # 減らさない
 
        X_min_up, y_min_up = resample(
            X_min, y_min,
            replace=True,
            n_samples=n_target,
            random_state=self.random_state,
        )

        X_res = np.vstack([X_maj, X_min_up])
        y_res = np.concatenate([y_maj, y_min_up])

        # シャッフル
        idx = np.random.default_rng(self.random_state).permutation(len(y_res))
        return self._make_result(
            X_train, y_train, X_res[idx], y_res[idx],
            params={"sampling_strategy": self.sampling_strategy},
        )


# ============================================================
# 3. ランダムアンダーサンプリング
# ============================================================

class RandomUnderSamplingStrategy(ImbalanceStrategy):
    """
    多数クラスのサンプルをランダムに「削除」して減らす。

    メリット  : 学習データが小さくなり高速化
    デメリット: 多数クラスの情報が失われる (情報損失)

    例え話: 正常取引のデータを間引いて、不正取引と件数を揃えるイメージ。
            便利だが、捨てたデータに重要な情報が入っていることもある。
    """

    def __init__(self, sampling_strategy: float = 0.1, random_state: int = 42) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "RandomUnderSampling (ランダム削除)"

    @property
    def category(self) -> str:
        return "undersampling"
    
    def apply(self, X_train: np.ndarray, y_train: np.ndarray) -> ResamplingResult:
        minority_mask = y_train == 1
        majority_mask = y_train == 0

        X_min = X_train[minority_mask]
        y_min = y_train[minority_mask]
        X_maj = X_train[majority_mask]
        y_maj = y_train[majority_mask]

        # 目標の多数クラス件数 (少数クラスに合わせて削減)
        n_minority = len(X_min)
        n_target = int(n_minority / self.sampling_strategy)
        n_target = min(n_target, len(X_maj))    # 増やさない

        X_maj_down, y_maj_down = resample(
            X_maj, y_maj,
            replace=False,
            n_samples=n_target,
            random_state=self.random_state,
        )

        X_res = np.vstack([X_maj_down, X_min])
        y_res = np.concatenate([y_maj_down, y_min])
 
        idx = np.random.default_rng(self.random_state).permutation(len(y_res))
        return self._make_result(
            X_train, y_train, X_res[idx], y_res[idx],
            params={"sampling_strategy": self.sampling_strategy},
        )


# ============================================================
# 4. SMOTE (Synthetic Minority Over-sampling Technique)
# ============================================================

class SmoteStrategy(ImbalanceStrategy):
    """
    少数クラスの「近傍サンプル間を補間」して合成データを生成する。

    RandomOverSampling との違い:
      複製(コピー)ではなく、既存サンプルの間を補間した「新しい」データを作る。
      -> 多様性があり、過学習しにくい。

    アルゴリズム:
      1. 少数クラスの各サンプルについて K近傍 を探す
      2. 近傍サンプルとの中間点をランダムに選んで合成データを生成

    例え話: 「不正取引A」と「不正取引B」の特徴の中間点に、
            架空の「不正取引AB」を新しく生み出すイメージ。
            コピーではなく「それっぽいデータ」を人工的に作る。
    """

    def __init__(
        self,
        sampling_strategy: float = 0.1,
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "SMOTE (合成少数サンプル生成)"

    @property
    def category(self) -> str:
        return "oversampling"

    def apply(self, X_train: np.ndarray, y_train: np.ndarray) -> ResamplingResult:
        rng = np.random.default_rng(self.random_state)

        minority_mask = y_train == 1
        majority_mask = y_train == 0

        X_min = X_train[minority_mask]
        y_min = y_train[minority_mask]
        X_maj = X_train[majority_mask]
        y_maj = y_train[majority_mask]

        n_majority = len(X_maj)
        n_target = int(n_majority * self.sampling_strategy)
        n_synthetic = max(n_target - len(X_min), 0)

        if n_synthetic == 0:
            return self._make_result(X_train, y_train, X_train, y_train)

        # K近傍を探す
        k = min(self.k_neighbors, len(X_min) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X_min)
        distances, indices = nn.kneighbors(X_min)

        # 合成サンプルの作成 
        synthetic_samples = []
        for _ in range(n_synthetic):
            # ランダムに基点サンプルを選ぶ
            base_idx = rng.integers(0, len(X_min))
            # 基点の K近傍からランダムに1つ選ぶ(自分自身[0]は除く)
            neighbor_local_idx = rng.integers(1, k+1)
            neighbor_idx = indices[base_idx][neighbor_local_idx]

            # 補間係数λ∈[0,1) をランダムに選ぶ 
            lam = rng.random()
            synthetic = X_min[base_idx] + lam * (X_min[neighbor_idx] - X_min[base_idx])
            synthetic_samples.append(synthetic)

        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.ones(n_synthetic, dtype=y_min.dtype)

        X_res = np.vstack([X_maj, X_min, X_synthetic])
        y_res = np.concatenate([y_maj, y_min, y_synthetic])

        idx = rng.permutation(len(y_res))
        return self._make_result(
            X_train, y_train, X_res[idx], y_res[idx],
            params = {
                "sampling_strategy": self.sampling_strategy,
                "k_neighbors": k,
                "n_synthetic_generated": n_synthetic,
            }
        )


# ============================================================
# 5. ADASYN (Adaptive Synthetic Sampling)
# ============================================================

class AdasynStrategy(ImbalanceStrategy):
    """
    SMOTEの発展版。「学習が難しいサンプル」ほど多く合成する。

    SMOTEとの違い: 
      SMOTEは少数クラス全体に均等に合成するが、
      ADASYNは「多数クラスに囲まれた少数クラス」のサンプルを 
      より多く生成することで、境界付近の学習を強化する。

    例え話: 試験問題で言えば、みんなが間違いやすい問題を
            重点的に練習問題として増やすイメージ。
    """

    def __init__(
        self,
        sampling_strategy: float = 0.1,
        k_neighbors: int = 5,
        random_state: int = 42,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "ADASYN (適応的合成サンプリング)"

    @property
    def category(self) -> str:
        return "oversampling"

    def apply(self, X_train: np.ndarray, y_train: np.ndarray) -> ResamplingResult:
        rng = np.random.default_rng(self.random_state)

        minority_mask = y_train == 1 
        majority_mask = y_train == 0
        
        X_min = X_train[minority_mask]
        y_min = y_train[minority_mask]
        X_maj = X_train[majority_mask]
        y_maj = y_train[majority_mask]

        n_majority = len(X_maj)
        n_target = int(n_majority * self.sampling_strategy)
        n_total_synthetic = max(n_target - len(X_min), 0)

        if n_total_synthetic == 0:
            return self._make_result(X_train, y_train, X_train, y_train)

        # 各少数サンプルの「難しさ」を計算 
        # -> K近傍の中で多数クラスが占める割合 r_i 
        k = min(self.k_neighbors, len(X_train)-1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X_train)
        _, indices = nn.kneighbors(X_min)

        r = np.array([
            (y_train[indices[i][1:]] == 0).sum() / k
            for i in range(len(X_min))
        ])

        # 正規化(合計が1になるよう)
        r_sum = r.sum()
        if r_sum == 0:
            # 全サンプルが均等(完全に分離されている場合)
            r_normalized = np.ones(len(X_min)) / len(X_min)
        else:
            r_normalized = r / r_sum 

        # 各サンプルに割り当てる合成数
        g = (r_normalized * n_total_synthetic).astype(int)
        # 端数調整
        shortage = n_total_synthetic - g.sum()
        if shortage > 0:
            top_indices = np.argsort(r_normalized)[::-1][:shortage]
            g[top_indices] += 1

        # 少数クラス内のK近傍で合成 
        nn_min = NearestNeighbors(n_neighbors=min(k + 1, len(X_min)), metric="euclidean")
        nn_min.fit(X_min)
        _, indices_min = nn_min.kneighbors(X_min)

        synthetic_samples = []
        for i, n_gen in enumerate(g):
            for _ in range(n_gen):
                neighbor_local = rng.integers(1, indices_min.shape[1])
                neighbor_idx = indices_min[i][neighbor_local]
                lam = rng.random()
                syn = X_min[i] + lam * (X_min[neighbor_idx] - X_min[i])
                synthetic_samples.append(syn)

        if not synthetic_samples:
            return self._make_result(X_train, y_train, X_train, y_train)
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.ones(len(synthetic_samples), dtype=y_min.dtype)

        X_res = np.vstack([X_maj, X_min, X_synthetic])
        y_res = np.concatenate([y_maj, y_min, y_synthetic])

        idx = rng.permutation(len(y_res))
        return self._make_result(
            X_train, y_train, X_res[idx], y_res[idx],
            params={
                "sampling_strategy": self.sampling_strategy,
                "k_neighbors": k,
                "n_synthetic_generated": len(synthetic_samples),
            },
        )


# ============================================================
# 6. 組み合わせ (Over + Under)
# ============================================================

class CombinedStrategy(ImbalanceStrategy):
    """
    オーバーサンプリングとアンダーサンプリングを組み合わせる。

    手順:
      1. 少数クラスをSMOTEで合成 (オーバーサンプリング)
      2. 多数クラスをランダムに削減 (アンダーサンプリング)

    メリット:
      - オーバーサンプリングのみより多数クラスの情報が整理される
      - アンダーサンプリングのみより少数クラスの多様性が増す 

    例え話: 不正取引の練習問題を増やしつつ(SMOTE)、
            正常取引の冗長なサンプルも間引く(Under)ことで
            学習データ全体のバランスを整えるイメージ。
    """

    def __init__(
        self,
        over_strategy: float = 0.1,
        under_strategy: float = 0.5,
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        over_strategy  : SMOTE で少数:多数をこの比率まで増やす
        under_strategy : その後、多数クラスをこの比率まで削減する
        """
        self.over_strategy = over_strategy
        self.under_strategy = under_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._smote = SmoteStrategy(over_strategy, k_neighbors, random_state)
        self._under = RandomUnderSamplingStrategy(under_strategy, random_state)

    @property
    def name(self) -> str:
        return "Combined (SMOTE + UnderSampling)"

    @property
    def category(self) -> str:
        return "combined"

    def apply(self, X_train: np.ndarray, y_train: np.ndarray) -> ResamplingResult:
        # Step1: SMOTE でオーバーサンプリング 
        over_result = self._smote.apply(X_train, y_train)
        # Step2: アンダーサンプリング 
        final_result = self._under.apply(
            over_result.X_resampled, over_result.y_resampled
        )
        # メタ情報は元データ基準で上書き
        return self._make_result(
            X_train, y_train,
            final_result.X_resampled, final_result.y_resampled,
            params={
                "over_strategy": self.over_strategy,
                "under_strategy": self.under_strategy,
                "k_neighbors": self.k_neighbors,
            },
        )


# ============================================================
# 7. クラス重み付け（サンプリングなし） 
# ============================================================

class ClassWeightStrategy(ImbalanceStrategy):
    """
    データを増減させず、「モデルの損失関数における重み」を調整する。

    考え方:
      不正取引を見逃した時のペナルティを、
      正常取引の誤検知ペナルティより大きく設定する。
      -> モデルが不正取引を重視して学習するようになる。

    注意:
      このクラスはデータを変形しない(X, y はそのまま返す)
      apply() で返す ResamplingResult のX_resampled は元データと同じ。
      モデル側に class_weight="balanced" or compute_class_weight で 
      算出した重みを渡す必要がある。

    例え話:
      テスト採点の配点を変えるイメージ。不正見逃し問題は10点、
      正常誤判定問題は1点にする、というような「重み付け採点」をモデルに課すイメージ。
    """

    @property
    def name(self) -> str:
        return "ClassWeight (クラス重み付け)"
    
    @property
    def category(self) -> str:
        return "algorithm"

    def apply(self, X_train: np.ndarray, y_train: np.ndarray) -> ResamplingResult:
        return self._make_result(X_train, y_train, X_train, y_train)

    def compute_weights(self, y_train: np.ndarray) -> dict:
        """
        sklearn の compute_class_weight に相当する重みを計算する。
        モデルの class_weight 引数に渡す用。
        
        Returns 
        ------- 
        {0: weight_majority, 1: weight_minority}
        """
        n_samples = len(y_train)
        n_classes = 2 
        n_maj = (y_train == 0).sum()
        n_min = (y_train == 1).sum()
        # sklearn の balanced 計算式: n_samples / (n_classes * n_k)
        w_maj = n_samples / (n_classes * n_maj)
        w_min = n_samples / (n_classes * n_min)

        return {0: w_maj, 1: w_min}


