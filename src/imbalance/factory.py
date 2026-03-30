"""
src/imbalance/factory.py 

ImbalanceStrategyFactory (Factory パターン)

役割:
  手法名(文字列)から対応するStrategyインスタンスを生成する。
  Notebookや実験スクリプトが具体クラスをimportせずすむ。

使用例:
  strategy = ImbalanceStrategyFactory.create("smote", sampling_strategy=0.1)
  result = strategy.apply(X_train, y_train)

  # 全手法を一括して比較実験
  all_strategies = ImbalanceStrategyFactory.create_all()
"""

#=================================================================
# Library
#=================================================================

# Standard
from itertools import starmap
from typing import List, Dict, Any 

# Custom
from .base import ImbalanceStrategy
from .strategies import (
    NoResamplingStrategy,
    RandomOverSamplingStrategy,
    RandomUnderSamplingStrategy,
    SmoteStrategy,
    AdasynStrategy,
    CombinedStrategy,
    ClassWeightStrategy,
)


#=================================================================
# 定数
#=================================================================

_STRATEGY_REGISTRY: Dict[str, type] = {
    "none": NoResamplingStrategy,
    "over": RandomOverSamplingStrategy,
    "under": RandomUnderSamplingStrategy,
    "smote": SmoteStrategy,
    "adasyn": AdasynStrategy,
    "combined": CombinedStrategy,
    "classweight": ClassWeightStrategy,
}


#=================================================================
# Codes 
#=================================================================

class ImbalanceStrategyFactory:
    """
    Strategy インスタンスを生成するFactoryクラス。
    全てのメソッドはクラスメソッドなのでインスタンス化不要。
    """

    @classmethod
    def create(
        cls,
        strategy_name: str,
        **kwargs: Any,
    ) -> ImbalanceStrategy:
        """
        手法名からStrategyインスタンスを生成する。

        Parameters
        ---------- 
        strategy_name: 手法名 [none|over|under|smote|adasyn|combined|classweight]
        **kwargs: 各メソッドごとのコンストラクタ引数

        Returns 
        ------- 
        ImbalanceStrategy サブクラスインスタンス 

        Raises 
        ------ 
        ValueError: 未登録の手法名だった場合
        """
        key = strategy_name.lower().strip()
        if key not in _STRATEGY_REGISTRY:
            available = list(_STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"未知の手法名: {strategy_name}\n"
                    f"使用可能な手法: {available}"
            )
        strategy_cls = _STRATEGY_REGISTRY[key]
        return strategy_cls(**kwargs)

    @classmethod
    def create_all(
        cls,
        sampling_strategy: float = 0.1,
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> List[ImbalanceStrategy]:
        """
        比較実験用に全手法のインスタンスをリストで返す。

        Parameters
        ---------- 
        sampling_strategy : オーバー/アンダーサンプリングの目標比率
        k_neighbors       : SMOTE / ADASYN の近傍数
        random_state      : 再現性のためのシード

        Returns
        ------- 
        List[ImbalanceStrategy]
        """
        common = dict(random_state=random_state)
        sampling = dict(sampling_strategy=sampling_strategy, **common)
        k_sampling = dict(k_neighbors=k_neighbors, **sampling)

        return [
            NoResamplingStrategy(),
            RandomOverSamplingStrategy(**sampling),
            RandomUnderSamplingStrategy(**sampling),
            SmoteStrategy(**k_sampling),
            AdasynStrategy(**k_sampling),
            CombinedStrategy(
                over_strategy=sampling_strategy,
                under_strategy=min(sampling_strategy * 5, 0.5),
                k_neighbors=k_neighbors,
                **common,
            ),
            ClassWeightStrategy(),
        ]

    @classmethod
    def available_strategies(cls) -> List[str]:
        """使用可能な手法名の一覧を返す"""
        return list(_STRATEGY_REGISTRY.keys())
    
    @classmethod
    def register(cls, name: str, strategy_cls: type) -> None:
        """
        カスタム戦略をレジストリに追加する(拡張ポイント)

        使用例:
          class MyCustomStrategy(ImbalanceStrategy)
          ImbalanceStrategyFactory.register("mycustom", MyCustomStrategy)
          s = ImbalanceStrategyFactory.create("mycustom")
        """
        if not issubclass(strategy_cls, ImbalanceStrategy):
            raise TypeError(
                f"{strategy_cls} は ImbalanceStrategy のサブクラスではありません"
            )
        _STRATEGY_REGISTRY[name.lower()] = strategy_cls
        print(f"✅ カスタム戦略 {name} を登録しました")
