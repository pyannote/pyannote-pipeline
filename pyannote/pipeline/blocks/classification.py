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

import warnings
import numpy as np
from typing import Optional

from ..pipeline import Pipeline
from ..parameter import Uniform
from pyannote.core.utils.distance import cdist
from pyannote.core.utils.distance import dist_range
from pyannote.core.utils.distance import l2_normalize


class ClosestAssignment(Pipeline):
    """Assign each sample to the closest target

    Parameters
    ----------
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'
    normalize : `bool`, optional
        L2 normalize vectors before clustering.

    Hyper-parameters
    ----------------
    threshold : `float`
        Do not assign if distance greater than `threshold`.
    """

    def __init__(self, metric: Optional[str] = 'cosine',
                       normalize: Optional[bool] = False):

        super().__init__()
        self.metric = metric
        self.normalize = normalize

        min_dist, max_dist = dist_range(metric=self.metric,
                                        normalize=self.normalize)
        if not np.isfinite(max_dist):
            # this is arbitray and might lead to suboptimal results
            max_dist = 1e6
            msg = (f'bounding distance threshold to {max_dist:g}: '
                   f'this might lead to suboptimal results.')
            warnings.warn(msg)
        self.threshold = Uniform(min_dist, max_dist)

    def __call__(self, X_target, X, method: Optional[str] = 'mean'):
        """Assign each sample to its closest class (if close enough)

        Parameters
        ----------
        X_target :
            Either : `np.ndarray`
                (n_targets, n_dimensions) target embeddings
            Or : iterable of n_targets
                all targets are `np.ndarray` which shape can differ:
                (m_samples, n_dimensions) target embeddings
        X : `np.ndarray`
            (n_samples, n_dimensions) sample embeddings
        method : `str`, optional
            Relevant iff the targets have several samples
            Used to reduce the distance matrix to a single float
            Available options (case-insensitive):
                - 'mean' (Default)
                - 'median'
                - 'min'

        Returns
        -------
        assignments : `np.ndarray`
            (n_samples, ) sample assignments
        """
        method=method.lower()
        if self.normalize:
            if isinstance(X_target,np.ndarray):
                X_target = l2_normalize(X_target)
            else:
                for i,target in enumerate(X_target):
                    X_target[i]= l2_normalize(target)
            X = l2_normalize(X)

        if isinstance(X_target,np.ndarray):
            distances = cdist(X_target, X, metric=self.metric)
        else:
            distances=[]
            for target in X_target:
                distance = cdist(target, X, metric=self.metric)
                if method=='mean':
                    distance=np.mean(distance)
                elif method=='median':
                    distance=np.median(distance)
                elif method=='min':
                    distance=np.min(distance)
                else:
                    raise ValueError(f"{method} is an invalid value for method, see \n{help(classifier)}")
                distances.append(distance)
        targets = np.argmin(distances, axis=0)

        # for i, k in enumerate(targets):
        #     if distances[k, i] > self.threshold:
        #         # do not assign
        #         targets[i] = -i

        return targets
