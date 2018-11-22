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

import warnings
import numpy as np
from typing import Optional

from scipy.cluster.hierarchy import fcluster
from pyannote.core.utils.hierarchy import linkage

import sklearn.cluster
from scipy.spatial.distance import squareform
from pyannote.core.utils.distance import pdist
from pyannote.core.utils.distance import dist_range
from pyannote.core.utils.distance import l2_normalize

import chocolate
from ..pipeline import Pipeline


class HierarchicalAgglomerativeClustering(Pipeline):
    """Hierarchical agglomerative clustering

    Parameters
    ----------
    method : `str`, optional
        Linkage method. Defaults to 'pool'.
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'
    normalize : `bool`, optional
        L2 normalize vectors before clustering.

    Hyper-parameters
    ----------------
    threshold : `float`
        Do not merge clusters with distance greater than `threshold`.
    """

    def __init__(self, method: Optional[str] = 'pool',
                       metric: Optional[str] = 'cosine',
                       normalize: Optional[bool] = False):

        super().__init__()
        self.method = method
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
        self.threshold = chocolate.uniform(min_dist, max_dist)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply hierarchical agglomerative clustering

        Parameters
        ----------
        X : `np.ndarray`
            (n_samples, n_dimensions) feature vectors.

        Returns
        -------
        y : `np.ndarray`
            (n_samples, ) cluster assignment (between 1 and n_clusters).
        """

        if self.normalize:
            X = l2_normalize(X)

        # compute agglomerative clustering all the way up to one cluster
        Z = linkage(X, method=self.method, metric=self.metric)

        # obtain flat clusters by applying distance threshold
        clusters = fcluster(Z, self.threshold, criterion='distance')

        return clusters


class AffinityPropagationClustering(Pipeline):
    """Clustering based on affinity propagation

    Parameters
    ----------
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'

    Hyper-parameters
    ----------------
    damping : `float`
    preference : `float`
        See `sklearn.cluster.AffinityPropagation`
    """

    def __init__(self, metric: Optional[str] = 'cosine'):

        super().__init__()
        self.metric = metric

        self.damping = chocolate.uniform(0.5, 1.0)
        self.preference = chocolate.uniform(-10., 0.)

    def instantiate(self):
        """Instantiate internal sklearn.cluster.AffinityPropagation"""

        self.affinity_propagation_ = sklearn.cluster.AffinityPropagation(
            damping=self.damping, preference=self.preference,
            affinity='precomputed', max_iter=200, convergence_iter=50)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply clustering based on affinity propagation

        Parameters
        ----------
        X : `np.ndarray`
            (n_samples, n_dimensions) feature vectors.

        Returns
        -------
        y : `np.ndarray`
            (n_samples, ) cluster assignment (between 1 and n_clusters).
        """

        n_samples, _ = X.shape

        try:
            affinity = -squareform(pdist(X, metric=self.metric))
            clusters = self.affinity_propagation_.fit_predict(affinity)
        except MemoryError as e:
            clusters = np.arange(n_samples)

        if np.any(clusters < 0):
            clusters = np.arange(n_samples)
        clusters += 1

        return clusters
