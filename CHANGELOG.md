### Version 1.1.1 (2019-04-09)

  - fix: do not raise FileExistsError when output directory exists in `pyannote-pipeline apply`
  - fix: skip evaluation of protocols without groundtruth in `pyannote-pipeline apply`
  - setup: switch to pyannote.database 2.1

### Version 1.1 (2019-03-20)

  - feat: add export to RTTM format
  - setup: switch to pyannote.database 2.0
  - fix: fix "use_threshold" parameter in HAC block

### Version 1.0 (2019-02-05)

  - feat: add support for pyannote.metrics (through `Pipeline.get_metric`)
  - feat: add support for optuna trial pruning
  - feat: keep track of processing & evaluation time

### Version 0.3 (2019-01-17)

  - feat: switch to optuna backend
  - feat: add "use_threshold" option to HAC pipeline
  - BREAKING: update Pipeline API
  - BREAKING: update Optimizer API
  - BREAKING: remove tensorboard support (for now)

### Version 0.2.1 (2018-12-04)

  - first public release
