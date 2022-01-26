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
import pytest

from pyannote.pipeline.parameter import ParamDict, Uniform, Integer, Frozen
from .utils import FakeTrial


def test_params_dict():
    params_dict = ParamDict(**{
        "param_a": Uniform(0, 1),
        "param_b": Integer(0, 10)
    })

    assert params_dict("params_dict", FakeTrial()) == {'param_a': 0.0,
                                                       'param_b': 0}


def test_dict_parameter_freezing():
    param_dict = ParamDict(param_a=Uniform(0, 1),
                           param_b=Integer(5, 10),
                           param_c=ParamDict(param_d=Uniform(0, 2),
                                             param_e=Uniform(0, 10)))
    param_dict.freeze({"param_b": 4, "param_c": {"param_d": 1}})
    _, params = zip(*param_dict.flatten().items())
    frozen_params = {param.value for param in params if isinstance(param, Frozen)}
    assert frozen_params == {4, 1}
