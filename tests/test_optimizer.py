#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2022 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Hadrien TITEUX

from typing import List, Dict, Any

import numpy as np
import pytest
from optuna.samplers import TPESampler

from pyannote.pipeline import Pipeline, Optimizer
from pyannote.pipeline.parameter import Integer, ParamDict
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

        def loss(self, data: float, y_preds: float) -> float:
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

        def loss(self, data: float, y_preds: float) -> float:
            return y_preds

        def get_direction(self) -> Direction:
            return direction

    optimizer_tester(pipeline=SumPipeline(), target=target)
