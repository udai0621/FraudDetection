"""
src/imbalance/base.py 

不均衡データ対処の抽象基底クラス (Strategyパターンの基底)

設計思想:
  全ての手法は「訓練データを受け取り、補正済みデータを返す」
  という共通インタフェースを持つ。
  呼び出し元 (Notebook / train.py) はどの手法を使っているかを意識しない。

クラス図:
  ImbalanceStrategy (ABC)
    |-- NoResamplingStrategy            # 何もしない(ベースライン)
    |-- RandomOverSamplingStrategy      # ランダムオーバーサンプリング
    |-- RandomUnderSamplingStrategy     # ランダムアンダーサンプリング
    |-- SmoteStrategy                   # SMOTE (合成少数サンプル生成)
    |-- AdasyncStrategy                 # ADASYN (適応的合成)
    |-- CombinedStrategy                # Over + Under 組み合わせ
    |-- ClassWeightStrategy             # クラス重み付け (サンプリングなし)
"""

#=================================================================
# Library
#=================================================================

# Standard 
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

# 3rd party 
import numpy as np 


#=================================================================
# Codes : Defines class
#=================================================================

@dataclass
class ResamplingResult:
    """
    各手法の適用結果をまとめるデータクラス。
    Notebook で結果を比較・可視化する時に使う。
    """
    strategy_name: str 
    X_resampled: np.ndarray 
    y_resampled: np.ndarray
    n_before_majority: int 
    n_before_minority: int 
    n_after_majority: int 
    n_after_minority: int 
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def imbalance_ratio_before(self) -> float:
        """補正前の多数:少数の比率"""
        return self.n_before_majority / max(self.n_before_minority, 1)

    @property
    def imbalance_ratio_after(self) -> float:
        """補正後の多数：少数の比率"""
        return self.n_after_majority / max(self.n_after_minority, 1)
    
    @property
    def minority_rate_before(self) -> float:
        total = self.n_before_majority + self.n_before_minority
        return self.n_before_minority / total

    @property
    def minority_rate_after(self) -> float:
        total = self.n_after_majority + self.n_after_minority
        return self.n_after_minority / total

    def summary(self) -> str:
        lines = [
            f"  戦略          : {self.strategy_name}",
            f"  補正前        : 多数={self.n_before_majority:,}  少数={self.n_before_minority:,}"
            f"  (少数率={self.minority_rate_before:.3%})",
            f"  補正後        : 多数={self.n_after_majority:,}  少数={self.n_after_minority:,}"
            f"  (少数率={self.minority_rate_after:.3%})",
            f"  不均衡比率    : {self.imbalance_ratio_before:.1f}:1 → {self.imbalance_ratio_after:.1f}:1",
        ]
        if self.params:
            lines.append(f"  パラメータ    : {self.params}")
        return "\n".join(lines)


class ImbalanceStrategy(ABC):
    """ 
    不均衡データ対処の抽象基底クラス (Strategy インタフェース)

    サブクラスは apply() を必ず実装する。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """手法の表示名"""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """
        カテゴリ分類:
          'oversampling'  : 少数クラスを増やす
          'undersampling' : 多数クラスを減らす 
          'combined'      : 両方
          'algorithm'     : モデル側で対処 (サンプリングなし)
        """
        ...

    @abstractmethod
    def apply(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ResamplingResult:
        """ 
        訓練データに不均衡補正を適用する。

        Parameters
        ---------- 
        X_train : shape (n_samples, n_features)
        y_train : shape (n_samples,) ※ 0=多数クラス, 1=少数クラス 

        Returns 
        ------- 
        ResamplingResult
        """
        ...

    def _make_result(
        self,
        X_before: np.ndarray,
        y_before: np.ndarray,
        X_after: np.ndarray,
        y_after: np.ndarray,
        params: Dict[str, Any] = None,
    ) -> ResamplingResult:
        """ResamplingResult を簡便に生成するヘルパー"""
        return ResamplingResult(
            strategy_name=self.name,
            X_resampled=X_after,
            y_resampled=y_after,
            n_before_majority=int((y_before == 0).sum()),
            n_before_minority=int((y_before == 1).sum()),
            n_after_majority=int((y_after == 0).sum()),
            n_after_minority=int((y_after == 1).sum()),
            params=params or {},
        )

