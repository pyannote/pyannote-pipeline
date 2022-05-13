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


import socket
import yaml
from pathlib import Path
import typer

from pyannote.database import get_protocol

from pyannote.core.utils.helper import get_class_by_name

from pyannote.pipeline import Pipeline
from pyannote.pipeline import Optimizer


def load_pipeline(config_yml: Path) -> Pipeline:

    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    # initialize pipeline
    pipeline_name = config["pipeline"]["name"]
    Klass = get_class_by_name(
        pipeline_name, default_module_name="pyannote.pipeline.blocks"
    )
    pipeline = Klass(**config["pipeline"].get("params", {}))

    # set frozen parameters
    pipeline.freeze(config.get("freeze", dict))

    # initialize preprocessors
    preprocessors = {}
    for key, preprocessor in config.get("preprocessors", {}).items():

        # preprocessors:
        #    key:
        #       name: package.module.ClassName
        #       params:
        #          param1: value1
        #          param2: value2
        if isinstance(preprocessor, dict):
            Klass = get_class_by_name(
                preprocessor["name"], default_module_name="pyannote.pipeline"
            )
            preprocessors[key] = Klass(**preprocessor.get("params", {}))
            continue

        try:
            # preprocessors:
            #    key: /path/to/database.yml
            preprocessors[key] = FileFinder(database_yml=preprocessor)

        except FileNotFoundError as e:
            # preprocessors:
            #    key: /path/to/{uri}.wav
            template = preprocessor
            preprocessors[key] = template

    return pipeline, preprocessors


app = typer.Typer()


@app.command()
def train(
    config: Path,
    protocol: str,
    subset: str = "development",
    sampler: str = "TPESampler",
    pruner: str = None,
    pretrained: Path = None,
):

    # load pipeline (and optional preprocessors)
    pipeline, preprocessors = load_pipeline(config)

    # gather training files
    p = get_protocol(protocol, preprocessors=preprocessors)
    files = list(getattr(p, subset)())

    work_dir = config.parent

    # using the hostname is useful because concurrent writing to sqlite db
    # does not work very well with filesystems shared by mulitple hosts.
    db = work_dir / f"{socket.gethostname()}.db"

    # the same sqlite DB can store trials optimized on different dataset x subset
    # TODO: store content of config.yml as an attribute of the study
    study_name = f"{protocol}.{subset}"

    optimizer = Optimizer(
        pipeline, db=db, study_name=study_name, sampler=sampler, pruner=pruner
    )

    if pretrained:
        pretrained_pipeline, _ = load_pipeline(pretrained)
        warm_start = pretrained_pipeline.parameters(frozen=True)
    else:
        warm_start = None

    trials = optimizer.tune_iter(files, warm_start=warm_start, show_progress=True)

    for trial in trials:
        pass


@app.command()
def apply(
    pretrained: Path, protocol: str, subset: str = "test",
):

    # load pipeline (and optional preprocessors)
    pipeline, preprocessors = load_pipeline(pretrained)
    pipeline.instantiate(pipeline.parameters(frozen=True))

    # gather test files
    p = get_protocol(protocol, preprocessors=preprocessors)
    files = list(getattr(p, subset)())

    for file in files:
        _ = pipeline(file)


if __name__ == "__main__":
    app()
