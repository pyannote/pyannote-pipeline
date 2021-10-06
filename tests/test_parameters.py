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
import pytest

from pyannote.pipeline.parameter import ParamDict, Uniform, Integer, ParamList, Categorical, Frozen
from .utils import FakeTrial


def test_params_dict():
    params_dict = ParamDict(**{
        "param_a": Uniform(0, 1),
        "param_b": Integer(0, 10)
    })

    assert params_dict("params_dict", FakeTrial()) == {'param_a': 0.0,
                                                       'param_b': 0}


def test_params_list():
    params_list = ParamList(*[
        Integer(i, 10) for i in range(5)
    ])
    assert params_list("params_list", FakeTrial()) == list(range(5))


def test_nested_params():
    nested = ParamList(*[
        ParamDict(**{
            "param_a": Uniform(0, 1),
            "param_b": Integer(0, 10)
        }),
        Categorical(["a", "b", "c"]),
        Uniform(0, 100)
    ])
    assert nested("nested_params", FakeTrial()) == [{'param_a': 0.0,
                                                     'param_b': 0},
                                                    'a', 0.0]

    nested = ParamDict(**{
        "param_list": ParamList(*[Uniform(i, 10) for i in range(3)]),
        "param_cat": Categorical(["a", "b", "c"])
    })

    assert nested("nested_params", FakeTrial()) == {'param_list': [0.0, 1.0, 2.0],
                                                    'param_cat': 'a'}


def test_dict_parameter_freezing():
    param_dict = ParamDict(param_a=Uniform(0, 1),
                           param_b=Integer(5, 10),
                           param_c=ParamDict(param_d=Uniform(0, 2),
                                             param_e=Uniform(0, 10)))
    param_dict.freeze({"param_b": 4, "param_c": {"param_d": 1}})
    _, params = zip(*param_dict.flatten().items())
    frozen_params = {param.value for param in params if isinstance(param, Frozen)}
    assert frozen_params == {4, 1}


def test_list_parameter_freezing():
    params_list = ParamList(*[
        ParamDict(param_a=Uniform(0, 2),
                  param_b=Uniform(0, 10))
        for _ in range(5)
    ])

    with pytest.raises(AssertionError):
        params_list.freeze([{"param_a": 1, } for _ in range(2)])
    params_list.freeze([{"param_a": 2} for _ in range(5)])
    _, params = zip(*params_list.flatten().items())
    frozen_params = [param.value for param in params if isinstance(param, Frozen)]
    assert frozen_params == [2] * 5
