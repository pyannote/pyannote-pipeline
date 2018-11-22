pyannote.pipeline
=================


# Installation

```bash
$ pip install --process-dependency-links pyannote.pipeline
```

If on MacOS, an `ghalton` fails to install, you might need to do something like

```bash
$ export CFLAGS="-Wno-deprecated-declarations -std=c++11 -stdlib=libc++"
$ pip install ghalton
```
