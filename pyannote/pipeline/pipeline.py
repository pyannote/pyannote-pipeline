#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

from typing import Optional
from typing import TextIO
from pathlib import Path
from collections import OrderedDict
import chocolate
from .typing import PipelineInput
from .typing import PipelineOutput

from filelock import FileLock
import yaml

from pyannote.core import Timeline
from pyannote.core import Annotation

class Pipeline:
    """Base pipeline"""

    def __init__(self):
        self._hyper_parameters = OrderedDict()
        self._frozen = OrderedDict()
        self._pipelines = OrderedDict()

    def __getattr__(self, name):

        if '_hyper_parameters' in self.__dict__:
            _hyper_parameters = self.__dict__['_hyper_parameters']
            if name in _hyper_parameters:
                return _hyper_parameters[name]

        if '_frozen' in self.__dict__:
            _frozen = self.__dict__['_frozen']
            if name in _frozen:
                return _frozen[name]

        if '_pipelines' in self.__dict__:
            _pipelines = self.__dict__['_pipelines']
            if name in _pipelines:
                return _pipelines[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        _hyper_parameters = self.__dict__.get('_hyper_parameters')
        _frozen = self.__dict__.get('_frozen')
        _pipelines = self.__dict__.get('_pipelines')

        # add/update one hyper-parameter
        if isinstance(value, chocolate.Distribution):
            if _hyper_parameters is None:
                raise AttributeError(
                    "cannot assign hyper-parameters before Pipeline.__init__() call")
            remove_from(self.__dict__, _frozen, _pipelines)
            _hyper_parameters[name] = value
            return

        # add/update one sub-pipeline
        if isinstance(value, Pipeline):
            if _pipelines is None:
                raise AttributeError(
                    "cannot assign pipelines before Pipeline.__init__() call")
            remove_from(self.__dict__, _hyper_parameters, _frozen)
            _pipelines[name] = value
            return

        # freeze hyper-parameter
        if _hyper_parameters is not None and name in _hyper_parameters:
            remove_from(_hyper_parameters)
            _frozen[name] = value
            return

        object.__setattr__(self, name, value)

    def __delattr__(self, name):

        if name in self._hyper_parameters:
            del self._hyper_parameters[name]

        elif name in self._pipelines:
            del self._pipelines[name]

        elif name in self._frozen:
            del self._frozen[name]

        else:
            object.__delattr__(self, name)

    @property
    def search_space(self):
        """Search space (used by pyannote.pipeline.Optimizer)"""

        space = OrderedDict()

        for sub_pipeline_name, sub_pipeline in self._pipelines.items():
            for param_name, param_value in sub_pipeline.search_space.items():
                space[f'{sub_pipeline_name}>{param_name}'] = param_value

        for param_name, param_value in self._hyper_parameters.items():
            space[param_name] = param_value

        return space

    @property
    def params(self):
        """Returns current set of parameters"""
        params = dict(self._hyper_parameters)
        params.update(self._frozen)
        for sub_pipeline_name, sub_pipeline in self._pipelines.items():
            params[sub_pipeline_name] = dict()
            for param_name, param_value in sub_pipeline.params.items():
                params[sub_pipeline_name][param_name] = param_value
        return params

    def instantiate(self):
        """Instantiate root pipeline with current set of hyper-parameters"""
        pass

    def freeze(self, params: dict) -> 'Pipeline':
        """Freeze pipeline parameters

        Parameters
        ----------
        params : `dict`
            Parameters.

        Returns
        -------
        self : `Pipeline`
            Pipeline with frozen parameters
        """

        _hyper_parameters = self.__dict__.get('_hyper_parameters')
        if _hyper_parameters is None:
            msg = "cannot freeze hyper-parameters before they are defined"
            raise AttributeError(msg)

        for name, value in params.items():

            # freeze sub-pipeline hyper-parameter
            if name in self._pipelines:

                if not isinstance(value, dict):
                    msg = (f"only hyper-parameters of '{name}' pipeline can "
                           f"be frozen (not the whole pipeline)")
                    raise ValueError(msg)

                self._pipelines[name].freeze(value)
                continue

            # freeze (or update frozen) root hyper-parameter
            if name in self._hyper_parameters or name in self._frozen:
                setattr(self, name, value)
                continue

            msg = f"hyper-parameter '{name}' does not exist"
            raise ValueError(msg)

        return self

    def with_params(self, params: dict) -> 'Pipeline':
        """Recursively instantiate all pipelines

        Parameters
        ----------
        params : `dict`
            Parameters

        Returns
        -------
        self : `Pipeline`
            Instantiated pipeline.
        """

        for name, value in params.items():

            # subpipeline
            if name in self._pipelines:
                if not isinstance(value, dict):
                    msg = (f"only hyper-parameters of '{name}' pipeline can "
                           f"be instantiated (not the whole pipeline)")
                    raise ValueError(msg)
                self._pipelines[name].with_params(value)
                continue

            # frozen hyper-parameters
            if name in self._frozen:
                if value != self._frozen[name]:
                    msg = (f"cannot change value of frozen hyper-parameter "
                           f"'{name}'")
                    raise ValueError(msg)
                continue

            # regular hyper-parameter
            if name in self._hyper_parameters:
                self._hyper_parameters[name] = value
                continue

            # anything else
            msg = f"hyper-parameter '{name}' does not exist"
            raise ValueError(msg)

        self.instantiate()

        return self

    def __call__(self, input: PipelineInput) -> PipelineOutput:
        """Apply pipeline on input and return its output"""
        raise NotImplementedError

    def loss(self, input: PipelineInput,
                   output: PipelineOutput) -> float:
        """Compute loss for given input/output pair"""
        raise NotImplementedError

    def write(self, file: TextIO,
                    output: PipelineOutput):
        """Write pipeline output to file"""

        if isinstance(output, Timeline):
            for s in output:
                file.write(f'{output.uri} {s.start:.3f} {s.end:.3f}\n')
            return

        if isinstance(output, Annotation):
            for s, t, l in output.itertracks(yield_label=True):
                file.write(f'{output.uri} {output.modality} {s.start:.3f} {s.end:.3f} {t} {l}\n')
            return

        raise NotImplementedError

    def unpack_params(self, params: dict) -> dict:
        """Unpack parameter dictionary

        param1                   param1
        param2                   param2
        sub_pipeline      <--
            sub_param1           sub_pipeline__sub_param1
            sub_param2           sub_pipeline__sub_param2

        Parameters
        ----------
        params : `dict`
            Flat dictionary of parameters.

        Returns
        -------
        unpacked_params : `dict`
            Nested dictionaries of parameters.
        """

        unpacked_params = {}

        pipeline_params = {name: {} for name in self._pipelines}
        for name, value in params.items():
            tokens = name.split('>')
            if len(tokens) > 1:
                pipeline_name = tokens[0]
                param_name = '>'.join(tokens[1:])
                pipeline_params[pipeline_name][param_name] = value
            else:
                unpacked_params[name] = value

        for name, pipeline in self._pipelines.items():
            unpacked_params[name] = pipeline.unpack_params(pipeline_params[name])

        return unpacked_params

    def dump_params(self, params_yml: Path,
                          params: Optional[dict] = None) -> str:
        """Dump parameters to disk

        Parameters
        ----------
        params_yml : `Path`
            Path to YAML file.
        params : `dict`, optional
            Parameters. Defaults to pipeline current parameters.

        Returns
        -------
        content : `str`
            Content written in `param_yml`.
        """

        if params is None:
            params = self.params
        content = yaml.dump(params, default_flow_style=False)
        with FileLock(params_yml.with_suffix('.lock')):
            with open(params_yml, mode='w') as fp:
                fp.write(content)
        return content

    def load_params(self, params_yml: Path) -> 'Pipeline':
        """Instantiate pipeline using parameters from disk

        Parameters
        ----------
        param_yml : `Path`
            Path to YAML file.

        Returns
        -------
        self : `Pipeline`
            Instantiated pipeline

        """

        with open(params_yml, mode='r') as fp:
            params = yaml.load(fp)
        return self.with_params(params)
