#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2021 CNRS

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

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform, Integer, ParamDict, ParamList, Categorical
from .utils import FakeTrial


def pipeline_tester(pl, fake_params):
    assert pl.parameters(FakeTrial()) == fake_params
    pl.instantiate(pl.parameters(FakeTrial()))
    assert (pl._unflatten(pl._flattened_parameters(instantiated=True))
            ==
            fake_params
            )


def test_pipeline_params_simple():
    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param_a = Uniform(0, 1)
            self.param_b = Integer(3, 10)

    pl = TestPipeline()
    # assert pl.parameters(FakeTrial()) == {"param_a": 0, "param_b": 3}
    pipeline_tester(pl, {"param_a": 0, "param_b": 3})


def test_pipeline_params_structured():
    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.params_dict = ParamDict(**{
                "param_a": Uniform(0, 1),
                "param_b": Integer(5, 10)
            })
            self.params_list = ParamList(*[
                Integer(i, 10) for i in range(5)
            ])

    pl = TestPipeline()
    fake_params = {'params_dict': {'param_a': 0.0,
                                   'param_b': 5},
                   'params_list': [0, 1, 2, 3, 4]}

    pipeline_tester(pl, fake_params)


def test_pipeline_with_subpipeline():
    class SubPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.param = Uniform(0, 1)

    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.params_dict = ParamDict(**{
                "param_a": Uniform(0, 1),
                "param_b": Integer(5, 10)
            })
            self.subpl = SubPipeline()

    pl = TestPipeline()
    fake_params = {'subpl': {'param': 0.0},
                   'params_dict': {'param_a': 0.0,
                                   'param_b': 5}
                   }

    pipeline_tester(pl, fake_params)


def test_pipeline_nested_parameters():
    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.nested_param = ParamList(*[
                ParamDict(**{
                    "param_a": Uniform(0, 1),
                    "param_b": Integer(5, 10)
                }),
                Categorical(["a", "b", "c"]),
                Uniform(50, 100)
            ])

    pl = TestPipeline()
    fake_params = {'nested_param': [{'param_a': 0.0, 'param_b': 5}, 'a', 50.0]}

    pipeline_tester(pl, fake_params)


def test_pipeline_freezing():
    class TestPipeline(Pipeline):

        def __init__(self):
            super().__init__()
            self.nested_param = ParamList(
                ParamDict(param_a=Uniform(0, 1),
                          param_b=Integer(5, 10),
                          param_c=ParamDict(param_d=Uniform(0, 2),
                                            param_e=Uniform(0, 10))),
                Categorical(["a", "b", "c"]),
                Uniform(50, 100)
            )

    pl = TestPipeline()
    frozen_params = {'nested_param': [{'param_b': 7, 'param_c': {'param_d': 2}}, "a", 70]}
    pl.instantiate(pl.parameters(FakeTrial()))
    pl.freeze(frozen_params)
    assert pl.parameters(frozen=True) == frozen_params
