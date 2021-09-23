from typing import Sequence

from optuna import Trial
from optuna.distributions import CategoricalChoiceType


class FakeTrial(Trial):

    def __init__(self):
        pass

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        return low

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        return low

    def suggest_categorical(self, name: str, choices: Sequence[CategoricalChoiceType]) -> CategoricalChoiceType:
        return choices[0]
