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

import time
import warnings
from pathlib import Path
from typing import Iterable, Optional, Callable, Generator, Union, Dict

import numpy as np
import optuna.logging
import optuna.pruners
import optuna.samplers
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import Trial, FixedTrial
from optuna.storages import RDBStorage, JournalStorage, JournalFileStorage
from tqdm import tqdm
from optuna.storages import RDBStorage, JournalStorage, JournalFileStorage

from .pipeline import Pipeline
from .typing import PipelineInput

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Optimizer:
    """Pipeline optimizer

    Parameters
    ----------
    pipeline : `Pipeline`
        Pipeline.
    db : `Path`, optional
        Path to trial database on disk. Use ".sqlite" extension for SQLite
        backend, and ".journal" for Journal backend (prefered for parallel
        optimization).
    study_name : `str`, optional
        Name of study. In case it already exists, study will continue from
        there. # TODO -- generate this automatically
    sampler : `str` or sampler instance, optional
        Algorithm for value suggestion. Must be one of "RandomSampler" or
        "TPESampler", or a sampler instance. Defaults to "TPESampler".
    pruner : `str` or pruner instance, optional
        Algorithm for early pruning of trials. Must be one of "MedianPruner" or
        "SuccessiveHalvingPruner", or a pruner instance.
        Defaults to no pruning.
    seed : `int`, optional
        Seed value for the random number generator of the sampler.
        Defaults to no seed.
    average_case : `bool`, optional
        Optimise for average case. Defaults to False (i.e. worst case).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        db: Optional[Path] = None,
        study_name: Optional[str] = None,
        sampler: Optional[Union[str, BaseSampler]] = None,
        pruner: Optional[Union[str, BasePruner]] = None,
        seed: Optional[int] = None,
        average_case: bool = True,
    ):
        self.pipeline = pipeline

        self.db = db
        if db is None:
            self.storage_ = None
        else:
            extension = Path(self.db).suffix
            if extension == ".db":
                warnings.warn(
                    "Storage with '.db' extension has been deprecated. Use '.sqlite' instead."
                )
                self.storage_ = RDBStorage(f"sqlite:///{self.db}")
            elif extension == ".sqlite":
                self.storage_ = RDBStorage(f"sqlite:///{self.db}")
            elif extension == ".journal":
                self.storage_ = JournalStorage(JournalFileStorage(f"{self.db}"))
        self.study_name = study_name

        if isinstance(sampler, BaseSampler):
            self.sampler = sampler
        elif isinstance(sampler, str):
            try:
                self.sampler = getattr(optuna.samplers, sampler)(seed=seed)
            except AttributeError as e:
                msg = '`sampler` must be one of "RandomSampler" or "TPESampler"'
                raise ValueError(msg)
        elif sampler is None:
            self.sampler = TPESampler(seed=seed)

        if isinstance(pruner, BasePruner):
            self.pruner = pruner
        elif isinstance(pruner, str):
            try:
                self.pruner = getattr(optuna.pruners, pruner)()
            except AttributeError as e:
                msg = '`pruner` must be one of "MedianPruner" or "SuccessiveHalvingPruner"'
                raise ValueError(msg)
        else:
            self.pruner = None

        # generate name of study based on pipeline hash
        # Klass = pipeline.__class__
        # study_name = f'{Klass.__module__}.{Klass.__name__}[{hash(pipeline)}]'

        self.study_ = optuna.create_study(
            study_name=self.study_name,
            load_if_exists=True,
            storage=self.storage_,
            sampler=self.sampler,
            pruner=self.pruner,
            direction=self.pipeline.get_direction(),
        )

        self.average_case = average_case

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

    def get_objective(
        self,
        inputs: Iterable[PipelineInput],
        show_progress: Union[bool, Dict] = False,
    ) -> Callable[[Trial], float]:
        """
        Create objective function used by optuna

        Parameters
        ----------
        inputs : `iterable`
            List of inputs to process.
        show_progress : bool or dict
            Show within-trial progress bar using tqdm progress bar.
            Can also be a **kwarg dict passed to tqdm.

        Returns
        -------
        objective : `callable`
            Callable that takes trial as input and returns correspond loss.
        """

        # this is needed for `inputs` that can be only iterated once.
        inputs = list(inputs)
        n_inputs = len(inputs)

        if show_progress == True:
            show_progress = {"desc": "Current trial", "leave": False, "position": 1}

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

            # use pyannote.metrics metric when available
            try:
                metric = self.pipeline.get_metric()
            except NotImplementedError as e:
                metric = None
                losses = []

            processing_time = []
            evaluation_time = []

            # instantiate pipeline with value suggested in current trial
            pipeline = self.pipeline.instantiate(self.pipeline.parameters(trial=trial))

            if show_progress != False:
                progress_bar = tqdm(total=len(inputs), **show_progress)
                progress_bar.update(0)

            # accumulate loss for each input
            for i, input in enumerate(inputs):
                # process input with pipeline
                # (and keep track of processing time)
                before_processing = time.time()
                output = pipeline(input)
                after_processing = time.time()
                processing_time.append(after_processing - before_processing)

                # evaluate output (and keep track of evaluation time)
                before_evaluation = time.time()

                # when metric is not available, use loss method instead
                if metric is None:
                    loss = pipeline.loss(input, output)
                    losses.append(loss)

                # when metric is available,`input` is expected to be provided
                # by a `pyannote.database` protocol
                else:
                    from pyannote.database import get_annotated

                    _ = metric(input["annotation"], output, uem=get_annotated(input))

                after_evaluation = time.time()
                evaluation_time.append(after_evaluation - before_evaluation)

                if show_progress != False:
                    progress_bar.update(1)

                if self.pruner is None:
                    continue

                trial.report(np.mean(losses) if metric is None else abs(metric), i)
                if trial.should_prune(i):
                    raise optuna.structs.TrialPruned()

            if show_progress != False:
                progress_bar.close()

            trial.set_user_attr("processing_time", sum(processing_time))
            trial.set_user_attr("evaluation_time", sum(evaluation_time))

            if metric is None:
                if len(np.unique(losses)) == 1:
                    mean = lower_bound = upper_bound = losses[0]
                else:
                    (mean, (lower_bound, upper_bound)), _, _ = bayes_mvs(
                        losses, alpha=0.9
                    )
            else:
                mean, (lower_bound, upper_bound) = metric.confidence_interval(alpha=0.9)

            if self.average_case:
                if metric is None:
                    return mean

                else:
                    return abs(metric)

            return (
                upper_bound
                if self.pipeline.get_direction() == "minimize"
                else lower_bound
            )

        return objective

    def tune(
        self,
        inputs: Iterable[PipelineInput],
        n_iterations: int = 10,
        warm_start: dict = None,
        show_progress: Union[bool, Dict] = True,
    ) -> dict:
        """Tune pipeline

        Parameters
        ----------
        inputs : iterable
            List of inputs processed by the pipeline at each iteration.
        n_iterations : int, optional
            Number of iterations. Defaults to 10.
        warm_start : dict, optional
            Nested dictionary of initial parameters used to bootstrap tuning.

        Returns
        -------
        result : dict
            ['loss']
            ['params'] nested dictionary of optimal parameters
        """

        # pipeline is currently being optimized
        self.pipeline.training = True

        objective = self.get_objective(inputs, show_progress=show_progress)

        if warm_start:
            flattened_params = self.pipeline._flatten(warm_start)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ExperimentalWarning)
                self.study_.enqueue_trial(flattened_params)

        self.study_.optimize(objective, n_trials=n_iterations, timeout=None, n_jobs=1)

        # pipeline is no longer being optimized
        self.pipeline.training = False

        return {"loss": self.best_loss, "params": self.best_params}

    def tune_iter(
        self,
        inputs: Iterable[PipelineInput],
        warm_start: dict = None,
        show_progress: Union[bool, Dict] = True,
    ) -> Generator[dict, None, None]:
        """

        Parameters
        ----------
        inputs : iterable
            List of inputs processed by the pipeline at each iteration.
        warm_start : dict, optional
            Nested dictionary of initial parameters used to bootstrap tuning.

        Yields
        ------
        result : dict
            ['loss']
            ['params'] nested dictionary of optimal parameters
        """

        objective = self.get_objective(inputs, show_progress=show_progress)

        try:
            best_loss = self.best_loss
        except ValueError as e:
            best_loss = np.inf

        if warm_start:
            flattened_params = self.pipeline._flatten(warm_start)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ExperimentalWarning)
                self.study_.enqueue_trial(flattened_params)

        while True:
            # pipeline is currently being optimized
            self.pipeline.training = True

            # one trial at a time
            self.study_.optimize(objective, n_trials=1, timeout=None, n_jobs=1)

            try:
                best_loss = self.best_loss
                best_params = self.best_params
            except ValueError as e:
                continue

            # pipeline is no longer being optimized
            self.pipeline.training = False

            yield {"loss": best_loss, "params": best_params}
