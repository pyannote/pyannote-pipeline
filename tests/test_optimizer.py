from typing import List, Dict, Any

import numpy as np
import pytest
from optuna.samplers import TPESampler

from pyannote.pipeline import Pipeline, Optimizer
from pyannote.pipeline.parameter import Integer, ParamList, ParamDict
from pyannote.pipeline.typing import Direction


def optimizer_tester(pipeline: Pipeline, target: Any):
    dataset = np.ones(10)
    sampler = TPESampler(seed=4577)
    optimizer = Optimizer(pipeline, sampler=sampler)
    optimizer.tune(dataset, n_iterations=100, show_progress=False)
    assert optimizer.best_params == target


@pytest.mark.parametrize("target, direction", [
    ({'param_a': 10, 'param_b': 10}, "maximize"),
    ({'param_a': 0, 'param_b': 0}, "minimize")
])
def test_basic_optimization(target, direction: Direction):
    class SumPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param_a: int = Integer(0, 10)
            self.param_b: int = Integer(0, 10)

        def __call__(self, data: float) -> float:
            return data + self.param_a + self.param_b

        def loss(self, dataset: np.ndarray, y_preds: float) -> float:
            return y_preds

        def get_direction(self) -> Direction:
            return direction

    optimizer_tester(pipeline=SumPipeline(), target=target)


@pytest.mark.parametrize("target, direction", [
    ({'list_param': [0, 1, 2]}, "maximize"),
    ({'list_param': [0, 0, 0]}, "minimize")
])
def test_structured_list_param_optim(target, direction: Direction):
    class SumPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.list_param: List[int] = ParamList([Integer(0, i) for i in range(3)])

        def __call__(self, data: float) -> float:
            return sum(self.list_param) + data

        def loss(self, dataset: np.ndarray, y_preds: float) -> float:
            return y_preds

        def get_direction(self) -> Direction:
            return direction

    optimizer_tester(pipeline=SumPipeline(), target=target)


@pytest.mark.parametrize("target, direction", [
    ({'param_dict': {'param_a': 10, 'param_b': 10}}, "maximize"),
    ({'param_dict': {'param_a': 0, 'param_b': 0}}, "minimize")
])
def test_structured_dict_param_optim(target, direction: Direction):
    class SumPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param_dict: Dict[str, int] = ParamDict(
                param_a=Integer(0, 10),
                param_b=Integer(0, 10)
            )

        def __call__(self, data: float) -> float:
            return data + self.param_dict["param_b"] + self.param_dict["param_a"]

        def loss(self, dataset: np.ndarray, y_preds: float) -> float:
            return y_preds

        def get_direction(self) -> Direction:
            return direction

    optimizer_tester(pipeline=SumPipeline(), target=target)


@pytest.mark.parametrize("target, direction", [
    ({'param_dict': {'param_a': 10, 'param_b': [2, 2]}}, "maximize"),
    ({'param_dict': {'param_a': 0, 'param_b': [0, 0]}}, "minimize")
])
def test_nested_structured_param_optim(target, direction: Direction):
    class SumPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param_dict = ParamDict(
                param_a=Integer(0, 10),
                param_b=ParamList([Integer(0, 2), Integer(0, 2)])
            )

        def __call__(self, data: float) -> float:
            return data + self.param_dict["param_a"] + sum(self.param_dict["param_b"])

        def loss(self, dataset: np.ndarray, y_preds: float) -> float:
            return y_preds

        def get_direction(self) -> Direction:
            return direction

    optimizer_tester(pipeline=SumPipeline(), target=target)
