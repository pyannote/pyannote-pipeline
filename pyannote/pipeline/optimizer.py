#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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

from typing import Iterable, Optional, Callable, Generator
from .typing import PipelineInput

import numpy as np

from pathlib import Path
from .pipeline import Pipeline

from optuna.trial import Trial, FixedTrial
import optuna.samplers
import optuna.pruners

import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Optimizer:
    """Pipeline optimizer

    Parameters
    ----------
    pipeline : `Pipeline`
        Pipeline.
    db : `Path`, optional
        Path to iteration database on disk.
    study_name : `str`, optional
        Name of study. In case it already exists, study will continue from
        there. # TODO -- generate this automatically
    sampler : `str`, optional
        Algorithm for value suggestion. Must be one of "RandomSampler" or
        "TPESampler". Defaults to no "TPESampler".
    pruner : `str`, optional
        Algorithm for early pruning of trials. Must be one of "MedianPruner" or
        "SuccessiveHalvingPruner". Defaults to no pruning.
    """

    def __init__(self, pipeline: Pipeline,
                       db: Optional[Path] = None,
                       study_name: Optional[str] = None,
                       sampler: Optional[str] = None,
                       pruner: Optional[str] = None):

        self.pipeline = pipeline

        self.db = db
        self.study_name = study_name

        self.sampler = "TPESampler" if sampler is None else sampler
        try:
            sampler = getattr(optuna.samplers, self.sampler)()
        except AttributeError as e:
            msg = '`sampler` must be one of "RandomSampler" or "TPESampler"'
            raise ValueError(msg)

        self.pruner = pruner
        if pruner is not None:
            try:
                pruner = getattr(optuna.pruners, self.pruner)()
            except AttributeError as e:
                msg = '`pruner` must be one of "MedianPruner" or "SuccessiveHalvingPruner"'
                raise ValueError(msg)

        # generate name of study based on pipeline hash
        # Klass = pipeline.__class__
        # study_name = f'{Klass.__module__}.{Klass.__name__}[{hash(pipeline)}]'

        self.study_ = optuna.create_study(
            study_name=self.study_name,
            load_if_exists=True,
            storage=f'sqlite:///{self.db}',
            sampler=sampler,
            pruner=pruner,
            direction='minimize')

    @property
    def best_loss(self) -> float:
        """Return best loss so far"""
        return self.study_.best_value

    @property
    def best_params(self) -> dict:
        """Return best parameters so far"""
        trial = FixedTrial(self.study_.best_params)
        return self.pipeline.parameters(trial=trial)

    @property
    def best_pipeline(self) -> Pipeline:
        """Return pipeline instantiated with best parameters so far"""
        return self.pipeline.instantiate(self.best_params)

    def get_objective(self, inputs: Iterable[PipelineInput]) -> \
        Callable[[Trial], float]:
        """
        Create objective function used by optuna

        Parameters
        ----------
        inputs : `iterable`
            List of inputs to process.

        Returns
        -------
        objective : `callable`
            Callable that takes trial as input and returns correspond loss.
        """

        # this is needed for `inputs` that can be only iterated once.
        inputs = list(inputs)

        def objective(trial: Trial) -> float:
            """Compute objective value

            Parameter
            ---------
            trial : `Trial`
                Current trial

            Returns
            -------
            loss : `float`
                Loss
            """

            # instantiate pipeline with value suggested in current trial
            pipeline = self.pipeline.instantiate(
                self.pipeline.parameters(trial=trial))

            # accumulate loss for each input
            losses = []
            for i, input in enumerate(inputs):
                output = pipeline(input)
                loss = pipeline.loss(input, output)
                losses.append(loss)

                if self.pruner is None:
                    continue

                # trial pruning
                trial.report(np.mean(losses), i)
                if trial.should_prune(i):
                    raise optuna.structs.TrialPruned()

            return np.mean(losses)

        return objective

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

        """

        objective = self.get_objective(inputs)
        self.study_.optimize(objective, n_trials=n_iterations,
                             timeout=None, n_jobs=1)

        return {'loss': self.best_loss,
                'params': self.best_params}

    def tune_iter(self, inputs: Iterable[PipelineInput]) -> \
        Generator[dict, None, None]:

        optuna.logging.get_verbosity()

        objective = self.get_objective(inputs)

        try:
            best_loss = self.best_loss
        except ValueError as e:
            best_loss = np.inf

        while True:

            # one trial at a time
            self.study_.optimize(objective, n_trials=1,
                                 timeout=None, n_jobs=1)

            try:
                best_loss = self.best_loss
                best_params = self.best_params
            except ValueError as e:
                continue

            yield {'loss': best_loss, 'params': best_params}
