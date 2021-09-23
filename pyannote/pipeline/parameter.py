#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2020 CNRS

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
# Hadrien TITEUX - https://github.com/hadware

from abc import ABCMeta, abstractmethod
from typing import Iterable, Any, Dict, List, Tuple

from optuna.trial import Trial


class Parameter(metaclass=ABCMeta):
    """Base hyper-parameter"""

    @abstractmethod
    def __call__(self, name: str, trial: Trial):
        pass


class StructuredParameter(Parameter):

    @abstractmethod
    def unflatten(self, flattened_params: Dict[str, Any]) -> Dict[str, Any]:
        pass


class ParamDict(StructuredParameter):
    """A mapping of parameters

    Parameters
    ----------
    params : dict of {string -> parameter} instance
        A dictionary of parameters
    """

    def __init__(self, params: Dict[str, Parameter]):
        super().__init__()
        for name, param in params.items():
            assert isinstance(name, str)
            assert isinstance(param, Parameter)
        self._params = params

    def __call__(self, name: str, trial: Trial):
        return {
            param_name: param(name, trial)
            for param_name, param
            in self._params.items()
        }

    def unflatten(self, flattened_params: Dict[str, Any]) -> Dict[str, Any]:
        params_dict = {}
        structured_params = {name: {} for name, param in self._params.items()
                             if isinstance(param, StructuredParameter)}
        for name, value in flattened_params.items():
            tokens = name.split(">")
            root_name: str = tokens[0]
            if len(tokens) > 1 and root_name in structured_params:
                subparam_name = ">".join(tokens[1:])
                structured_params[root_name][subparam_name] = value
            else:
                params_dict[name] = value

        # recursively unflatten structured parameter flattened dictionary
        for name, param in self._params.items():
            if not isinstance(param, StructuredParameter):
                continue
            params_dict[name] = param.unflatten(structured_params[name])

        return params_dict


class ParamList(StructuredParameter):
    """A list of parameters

    Parameters
    ----------
    params : list of parameters instance
        A list of parameters
    """

    def __init__(self, params: List[Parameter]):
        super().__init__()
        for param in params:
            assert isinstance(param, Parameter)
        self.params = params

    def __call__(self, name: str, trial: Trial):
        return [param(name, trial) for param in self.params]

    def unflatten(self, flattened_params: Dict[str, Any]) -> List[Any]:
        params_list: List[Tuple[int, Any]] = []
        structured_params = {idx: {} for idx, param in enumerate(self.params)
                             if isinstance(param, StructuredParameter)}
        params_indices = []
        for name, value in flattened_params.items():
            tokens = name.split(">")
            root_name = tokens[0]
            assert root_name.isdigit()
            param_idx: int = int(root_name)
            params_indices.append(param_idx)

            if len(tokens) > 1:
                assert param_idx in structured_params
                subparam_name = ">".join(tokens[1:])
                structured_params[param_idx][subparam_name] = value
            else:
                params_list.append((param_idx, value))

        # recursively unflatten structured parameter flattened dictionary
        for idx, param in enumerate(self.params):
            if not isinstance(param, StructuredParameter):
                continue
            params_list.append((idx, param.unflatten(structured_params[idx])))

        # sorting by idx
        all_idx, params = zip(*sorted(params_list, key=lambda x: x[0]))

        # checking that we have the same amount of parameters in the instance
        # as we have in the flattened param input
        assert all_idx == list(range(len(self.params)))

        return params


class Categorical(Parameter):
    """Categorical hyper-parameter

    The value is sampled from `choices`.

    Parameters
    ----------
    choices : iterable
        Candidates of hyper-parameter value.
    """

    def __init__(self, choices: Iterable):
        super().__init__()
        self.choices = list(choices)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_categorical(name, self.choices)


class DiscreteUniform(Parameter):
    """Discrete uniform hyper-parameter

    The value is sampled from the range [low, high],
    and the step of discretization is `q`.

    Parameters
    ----------
    low : `float`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `float`
        Upper endpoint of the range of suggested values.
        `high` is included in the range.
    q : `float`
        A step of discretization.
    """

    def __init__(self, low: float, high: float, q: float):
        super().__init__()
        self.low = float(low)
        self.high = float(high)
        self.q = float(q)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_discrete_uniform(name, self.low, self.high, self.q)


class Integer(Parameter):
    """Integer hyper-parameter

    The value is sampled from the integers in [low, high].

    Parameters
    ----------
    low : `int`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `int`
        Upper endpoint of the range of suggested values.
        `high` is included in the range.
    """

    def __init__(self, low: int, high: int):
        super().__init__()
        self.low = int(low)
        self.high = int(high)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_int(name, self.low, self.high)


class LogUniform(Parameter):
    """Log-uniform hyper-parameter

    The value is sampled from the range [low, high) in the log domain.

    Parameters
    ----------
    low : `float`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `float`
        Upper endpoint of the range of suggested values.
        `high` is excluded from the range.
    """

    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = float(low)
        self.high = float(high)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_loguniform(name, self.low, self.high)


class Uniform(Parameter):
    """Uniform hyper-parameter

    The value is sampled from the range [low, high) in the linear domain.

    Parameters
    ----------
    low : `float`
        Lower endpoint of the range of suggested values.
        `low` is included in the range.
    high : `float`
        Upper endpoint of the range of suggested values.
        `high` is excluded from the range.
    """

    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = float(low)
        self.high = float(high)

    def __call__(self, name: str, trial: Trial):
        return trial.suggest_uniform(name, self.low, self.high)


class Frozen(Parameter):
    """Frozen hyper-parameter

    The value is fixed a priori

    Parameters
    ----------
    value :
        Fixed value.
    """

    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    def __call__(self, name: str, trial: Trial):
        return self.value
