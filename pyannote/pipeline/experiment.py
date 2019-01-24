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

"""
Pipeline

Usage:
  pyannote-pipeline train [options] [(--forever | --iterations=<iterations>)] <experiment_dir> <database.task.protocol>
  pyannote-pipeline best [options] <experiment_dir> <database.task.protocol>
  pyannote-pipeline apply [options] <params.yml> <database.task.protocol> <output_dir>
  pyannote-pipeline -h | --help
  pyannote-pipeline --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset. Defaults to 'development' in "train"
                             mode, and to all subsets in "apply" mode.
"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  --iterations=<iterations>  Number of iterations. [default: 1]
  --forever                  Iterate forever.
  --sampler=<sampler>        Choose sampler between RandomSampler or TPESampler
                             [default: TPESampler].
  --pruner=<pruner>          Choose pruner between MedianPruner or
                             SuccessiveHalvingPruner. Defaults to no pruning.

"apply" mode:
  <params.yml>               Path to hyper-parameters.
  <output_dir>               Directory where to store pipeline output.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml that describes the pipeline.

    ................... <experiment_dir>/config.yml ...................
    pipeline:
       name: Yin2018
       params:
          sad: tutorials/pipeline/sad
          scd: tutorials/pipeline/scd
          emb: tutorials/pipeline/emb
          metric: angular

    # preprocessors can be used to automatically add keys into
    # each (dict) file obtained from pyannote.database protocols.
    preprocessors:
       audio: ~/.pyannote/db.yml   # load template from YAML file
       video: ~/videos/{uri}.mp4   # define template directly

    # one can freeze some hyper-parameters if needed (e.g. when
    # only part of the pipeline needs to be updated)
    freeze:
       speech_turn_segmentation:
          speech_activity_detection:
              onset: 0.5
              offset: 0.5
    ...................................................................

"train" mode:
    Tune the pipeline hyper-parameters
        <experiment_dir>/<database.task.protocol>.<subset>.yml

"best" mode:
    Display current best loss and corresponding hyper-paramters.

"apply" mode
    Apply the pipeline (with best set of hyper-parameters)

"""

import os
import yaml
import numpy as np
from typing import Optional
from pathlib import Path
from docopt import docopt

import itertools
from tqdm import tqdm

from pyannote.database import FileFinder
from pyannote.database import get_protocol

from pyannote.core.utils.helper import get_class_by_name
from .optimizer import Optimizer


class Experiment:
    """Pipeline experiment

    Parameters
    ----------
    experiment_dir : `Path`
        Experiment root directory.
    training : `bool`, optional
        Switch to training mode
    """

    CONFIG_YML = '{experiment_dir}/config.yml'
    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'

    @classmethod
    def from_params_yml(cls, params_yml: Path,
                             training : bool = False) -> 'Experiment':
        """Load pipeline from 'params.yml' file

        Parameters
        ----------
        params_yml : `Path`
            Path to 'params.yml' file.
        training : `bool`, optional
            Switch to training mode.

        Returns
        -------
        xp : `Experiment`
            Pipeline experiment.
        """

        experiment_dir = params_yml.parents[2]
        xp = cls(experiment_dir, training=training)
        xp.pipeline_.load_params(params_yml)
        return xp

    def __init__(self, experiment_dir: Path, training: bool = False):

        super().__init__()

        self.experiment_dir = experiment_dir

        # load configuration file
        config_yml = self.CONFIG_YML.format(experiment_dir=self.experiment_dir)
        with open(config_yml, 'r') as fp:
            self.config_ = yaml.load(fp)

        # initialize preprocessors
        preprocessors = {}
        for key, db_yml in self.config_.get('preprocessors', {}).items():
            try:
                preprocessors[key] = FileFinder(db_yml)
            except FileNotFoundError as e:
                template = db_yml
                preprocessors[key] = template
        self.preprocessors_ = preprocessors

        # initialize pipeline
        pipeline_name = self.config_['pipeline']['name']
        Klass = get_class_by_name(
            pipeline_name, default_module_name='pyannote.audio.pipeline')
        self.pipeline_ = Klass(**self.config_['pipeline'].get('params', {}))

        # freeze  parameters
        if 'freeze' in self.config_:
            params = self.config_['freeze']
            self.pipeline_.freeze(params)

    def train(self, protocol_name: str,
                    subset: Optional[str] = 'development',
                    n_iterations: Optional[int] = 1,
                    sampler: Optional[str] = None,
                    pruner: Optional[str] = None):
        """Train pipeline

        Parameters
        ----------
        protocol_name : `str`
            Name of pyannote.database protocol to use.
        subset : `str`, optional
            Use this subset for training. Defaults to 'development'.
        n_iterations : `int`, optional
            Number of iterations. Defaults to 1.
        sampler : `str`, optional
            Choose sampler between RandomSampler and TPESampler
        pruner : `str`, optional
            Choose between MedianPruner or SuccessiveHalvingPruner.
        """
        train_dir = Path(self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset))
        train_dir.mkdir(parents=True, exist_ok=True)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        study_name = "default"
        optimizer = Optimizer(self.pipeline_,
                              db=train_dir / 'iterations.db',
                              study_name=study_name,
                              sampler=sampler,
                              pruner=pruner)

        params_yml = train_dir / 'params.yml'

        progress_bar = tqdm(unit='iteration')
        progress_bar.set_description('Waiting for first iteration to complete')
        progress_bar.update(0)

        inputs = list(getattr(protocol, subset)())
        iterations = optimizer.tune_iter(inputs)

        try:
            best_loss = optimizer.best_loss
        except ValueError as e:
            best_loss = np.inf
        count = itertools.count() if n_iterations < 0 else range(n_iterations)

        for i, status in zip(count, iterations):

            best_loss = status['loss']
            best_params = status['params']

            self.pipeline_.dump_params(params_yml, params=best_params)

            # progress bar
            desc = f'Best = {100 * best_loss:g}%'
            progress_bar.set_description(desc=desc)
            progress_bar.update(1)

    def best(self, protocol_name: str,
                   subset: str = 'development'):
        """Print current best pipeline

        Parameters
        ----------
        protocol_name : `str`
            Name of pyannote.database protocol used for training.
        subset : `str`, optional
            Subset used for training. Defaults to 'development'.
        """

        train_dir = Path(self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset))

        study_name = "default"
        optimizer = Optimizer(self.pipeline_,
                              db=train_dir / 'iterations.db',
                              study_name=study_name)

        try:
            best_loss = optimizer.best_loss
        except ValueError as e:
            print('Still waiting for at least one iteration to succeed.')
            return

        best_params = optimizer.best_params

        print(
            f'Loss = {100 * best_loss:g}% '
            f'with the following hyper-parameters:')

        content = yaml.dump(best_params, default_flow_style=False)
        print(content)

    def apply(self, protocol_name: str,
                    output_dir: Path,
                    subset: Optional[str] = None):
        """Apply current best pipeline

        Parameters
        ----------
        protocol_name : `str`
            Name of pyannote.database protocol to process.
        subset : `str`, optional
            Subset to process. Defaults processing all subsets.
        """

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        output_dir.mkdir(parents=True, exist_ok=False)
        if subset is None:
            path = output_dir / f'{protocol_name}.all.txt'
        else:
            path = output_dir / f'{protocol_name}.{subset}.txt'

        # initialize evaluation metric
        try:
            metric = self.pipeline_.get_metric()
        except NotImplementedError as e:
            metric = None
            losses = []

        with open(path, mode='w') as fp:

            if subset is None:
                files = FileFinder.protocol_file_iter(protocol)
            else:
                files = getattr(protocol, subset)()

            for current_file in files:
                output = self.pipeline_(current_file)

                # evaluate output
                if metric is None:
                    loss = self.pipeline_.loss(current_file, output)
                    losses.append(loss)

                else:
                    from pyannote.database import get_annotated
                    _ = metric(input['annotation'], output,
                               uem=get_annotated(current_file))

                self.pipeline_.write(fp, output)

        # report evaluation metric
        if metric is None:
            loss = np.mean(losses)
            print(f'Loss = {loss:g}')
        else:
            _ = metric.report(display=True)

def main():

    arguments = docopt(__doc__, version='Tunable pipelines')

    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['train']:

        if subset is None:
            subset = 'development'

        if arguments['--forever']:
            iterations = -1
        else:
            iterations = int(arguments['--iterations'])

        sampler = arguments['--sampler']
        pruner = arguments['--pruner']

        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        experiment = Experiment(experiment_dir, training=True)
        experiment.train(protocol_name, subset=subset, n_iterations=iterations,
                         sampler=sampler, pruner=pruner)

    if arguments['best']:

        if subset is None:
            subset = 'development'

        experiment_dir = Path(arguments['<experiment_dir>'])
        experiment_dir = experiment_dir.expanduser().resolve(strict=True)

        experiment = Experiment(experiment_dir, training=False)
        experiment.best(protocol_name, subset=subset)

    if arguments['apply']:

        params_yml = Path(arguments['<params.yml>'])
        params_yml = params_yml.expanduser().resolve(strict=True)

        output_dir = Path(arguments['<output_dir>'])
        output_dir = output_dir.expanduser().resolve(strict=False)

        experiment = Experiment.from_params_yml(params_yml, training=False)
        experiment.apply(protocol_name, output_dir, subset=subset)
