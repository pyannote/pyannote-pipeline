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

from typing import Iterable, Generator, Optional
from .typing import PipelineInput

import numpy as np

from pathlib import Path
from .pipeline import Pipeline

import chocolate
from chocolate.base import SearchAlgorithm
from chocolate import SQLiteConnection

class Optimizer:
    """Pipeline optimizer

    Parameters
    ----------
    pipeline : `Pipeline`
        Pipeline.
    db : `Path`
        Path to iteration database on disk.
    sampler : `str`, optional
        One of "Random", "QuasiRandom", "Bayes", or "CMAES".
        Defaults to "QuasiRandom".
    """

    def __init__(self, pipeline: Pipeline,
                       db: Path,
                       sampler: Optional[str] = "QuasiRandom"):

        self.pipeline = pipeline

        self.db = db
        self.connection_ = SQLiteConnection(f'sqlite:///{self.db}')

        self.sampler = sampler

        try:
            Sampler = getattr(chocolate, sampler)
        except AttributeError as e:
            msg = ('"sampler must be one of "Random", "QuasiRandom", '
                   '"Bayes", or "CMAES".')
            raise ValueError(msg)

        if not issubclass(Sampler, SearchAlgorithm):
            msg = ('"sampler must be one of "Random", "QuasiRandom", '
                   '"Bayes", or "CMAES".')
            raise ValueError(msg)

        self.sampler_ = Sampler(self.connection_,
                                self.pipeline.search_space)

    @property
    def best_params(self) -> dict:
        """Return best parameters so far"""
        status = self.status
        if 'best' not in status:
            msg = 'cannot get parameters as no trial has finished yet.'
            raise AttributeError(msg)
        return status['best']['params']

    @property
    def best_pipeline(self) -> Pipeline:
        """Return pipeline instantiated with best parameters so far"""
        return self.pipeline.with_params(self.best_params)

    @property
    def status(self) -> dict:
        """Return current status of optimizer

        Returns
        -------
        status : `dict`
            ['iterations']['count'] (`int`) total number of iterations
            ['iterations']['done'] (`int`) total number of sucessful iterations
            ['best']['loss'] (`float`) best loss so far
            ['best']['params'] (`dict`) best parameters so far

        Note
        ----
        If no iteration has finished, `status` does not provide a 'best' key.
        """

        iterations = self.connection_.all_results()

        # total number of iterations (including those which failed and those
        # currently running)
        count = len(iterations)

        # total number of successful iterations
        done = 0
        for iteration in iterations:
            loss = iteration.get('_loss')
            if loss is None or not np.isfinite(loss):
                continue
            done += 1

        status = {'iterations': {'count': count, 'done': done}}
        if done < 1:
            return status

        def get_loss(iteration):
            loss = iteration.get('_loss')
            if loss is None:
                return np.inf
            return loss

        best_iteration = min(iterations, key=get_loss)
        del best_iteration['id']
        del best_iteration['_chocolate_id']
        loss = best_iteration.pop('_loss')

        params = {str(name): value
                  for name, value in best_iteration.items()}
        # chocolate stores parameters as a flat dictionary
        # while pipeline expects nested dictionaries
        params = self.pipeline.unpack_params(params)
        status['best'] = {'loss': loss, 'params': params}

        return status

    def tune_iter(self, inputs: Iterable[PipelineInput]) -> Generator[dict, None, None]:
        """Tune pipeline forever

        Parameters
        ----------
        inputs : iterable
            List of inputs processed by the pipeline at each iteration.

        Yields
        ------
        status : dict
            ['iteration'] (`int`) latest iteration index
            ['loss'] (`float`) latest iteration loss
            ['params'] (`dict`) latest iteration parameters
            ['new_best'] (`bool`) whether latest iteration is the best so far
        """

        status = self.status
        best_loss = status.get('best', {'loss': np.inf})['loss']

        while True:

            # get next set of hyper-parameters to try
            token, params = self.sampler_.next()

            # depending on the sampler, params maybe float or np.float
            params = {name: value.item() if hasattr(value, 'item') else value
                      for name, value in params.items()}

            # chocolate stores parameters as a flat dictionary
            # while pipeline expects nested dictionaries
            params = self.pipeline.unpack_params(params)

            # instantiate pipeline with this set of parameters
            self.pipeline.with_params(params)

            # NOTE this is embarrasingly parallel. do something about this
            losses = []
            for input in inputs:
                output = self.pipeline(input)
                loss = self.pipeline.loss(input, output)
                losses.append(loss)
            loss = sum(losses) / len(losses)

            result = {'iteration': token['_chocolate_id'] + 1,
                      'loss': loss,
                      'params': params,
                      'new_best': False}

            if loss is not None and loss < best_loss:

                # update best loss from disk as another process might have
                # already reached a better loss in the meantime
                best_loss = self.status.get('best', {'loss': np.inf})['loss']

                if loss < best_loss:
                    result['new_best'] = True
                    best_loss = loss

            self.sampler_.update(token, loss)
            yield result


    def tune(self, inputs: Iterable[PipelineInput],
                   n_iterations: int=10) -> dict:
        """Tune pipeline

        Parameters
        ----------
        inputs : iterable
            List of inputs processed by the pipeline at each iteration.
        n_iterations : int, optional
            Number of iterations. Defaults to 10.

        Returns
        -------
        status : `dict`
            ['iterations']['count'] (`int`) total number of iterations
            ['iterations']['done'] (`int`) total number of sucessful iterations
            ['best']['loss'] (`float`) best loss
            ['best']['params'] (`dict`) best parameters

        Note
        ----
        If no iteration has completed successfully, `status` does not contain
        a 'best' key.
        """

        iterations = self.tune_iter(inputs)

        for i in range(n_iterations):
            _ = next(iterations)

        return self.status
