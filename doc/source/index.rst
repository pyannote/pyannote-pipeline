
#################
pyannote.pipeline
#################

`pyannote.pipeline` is an open-source Python library for pipeline optimization


Installation
============

::

$ conda create -n pyannote python=3.6 anaconda
$ source activate pyannote
$ pip install --process-dependency-links pyannote.pipeline

If on MacOS, an `ghalton` fails to install, you might need to do something like

::

$ export CFLAGS="-Wno-deprecated-declarations -std=c++11 -stdlib=libc++"
$ pip install ghalton
