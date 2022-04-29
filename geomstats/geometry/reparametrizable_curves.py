"""Reparameterizable curves on any given manifold.

Lead author: Adel Ardalan.
"""

import math
from typing import Dict

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.manifold import Manifold

R2 = Euclidean(dim=2)
R3 = Euclidean(dim=3)


class ReparametrizableCurve(Manifold):
    r"""A reparametrizable discrete curve.

    A reparametrizable discrete curve characterized by one parameter
    and the corresponding points in ambient_manifold. Each curve is represented
     by a dict/list of tuples of the form `(param, point)` where `param` is
     either __time__ or __arclength__ and `point` is is the corresponding point
     on the curve in the `ambient_manifold`.

    Parameters
    ----------
    points : Dict or List
        Dict/list of points on the curve.
    param_type : str
        The type of parameter used for the points. Can be `'t'` (for time) or
        `'s'` (for arclength).
    ambient_manifold : Manifold
        Manifold in which curves take values.

    Attributes
    ----------
    t_curve : Dict
        Sorted dict mapping `'t'` values to points on the curve.
    s_curve : Dict
        Sorted dict mapping `'s'` values to points on the curve.
    ambient_manifold : Manifold
        Manifold in which curves take values.
    """

    def __init__(
        self, points: Dict, param_type: str, ambient_manifold: Manifold, **kwargs
    ):
        super(ReparametrizableCurve, self).__init__(
            dim=math.inf, shape=(), default_point_type="vector", **kwargs
        )
        self.t_curve, self.s_curve = {}, {}
        srt_params = sorted(list(points.keys()))
        if srt_params[0] != 0:
            raise ValueError(
                f"First parameter value should be zero; got {srt_params[0]}."
            )

        if param_type == "t":
            sum_s = 0
            for idx, kk in enumerate(srt_params):
                self.t_curve[kk] = points[kk]
                if idx > 0:
                    sum_s += np.linalg.norm(
                        self.t_curve[kk] - self.t_curve[srt_params[idx - 1]]
                    )
                self.s_curve[sum_s] = points[kk]
        elif param_type == "s":
            sum_t = 0
            curve_length = sum(
                [
                    np.linalg.norm(
                        self.t_curve[srt_params[idx]]
                        - self.t_curve[srt_params[idx - 1]]
                    )
                    for idx in range(1, len(srt_params))
                ]
            )
            for idx, kk in enumerate(srt_params):
                self.s_curve[kk] = points[kk]
                if idx > 0:
                    sum_t += np.linalg.norm(
                        self.s_curve[kk] - self.s_curve[srt_params[idx - 1]]
                    )
                self.t_curve[sum_s / curve_length] = points[kk]
        else:
            raise ValueError(f"Invalid parameter type {param_type}.")
        self.ambient_manifold = ambient_manifold

    def curvatures(self, max_order=None, atol=gs.atol, **kwargs):
        """Comuptes curvatures of all orders up to max_order.

        Computes the generalized curvature values based on numerical
        derivatives, a qr factorization, and the standard formula.

        Code converted from Matlab as per http://bitly.ws/qxkU

        Parameters
        ----------
        max_order : int
            Max order of curvatures to be computed.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        curvatures : Dict
            Sorted dict mapping `'s'` values to curvatures of different
            orders at various points on the curve.
        """
        if max_order is None:
            max_order = self.ambient_manifold.dim - 1

        smooth_curvatures = None
        if "smooth_curvatures" in kwargs:
            smooth_curvatures = kwargs["smooth_curvatures"]
            if smooth_curvatures["method"] == "boxcar":
                smooth_curvatures_ws = smooth_curvatures["win_width"]
                smooth_curvatures_kernel = (
                    np.ones(smooth_curvatures_ws) / smooth_curvatures_ws
                )
            else:
                raise ValueError(
                    f'Invalid smoothing method {smooth_curvatures["method"]}.'
                )

        s_points_T = np.array(list(self.s_curve.values())).T

        if smooth_curvatures is not None:
            for idx in range(s_points_T.shape[0]):
                s_points_T[idx, :] = np.convolve(
                    s_points_T[idx, :], smooth_curvatures_kernel, mode="same"
                )

        nd = np.zeros((s_points_T.shape[0], s_points_T.shape[1], max_order + 2))
        nd[:, :, 0] = s_points_T

        for jj in range(1, max_order + 1):
            for ii in range(1, s_points_T.shape[1] - 2):
                nd[:, ii, jj] = nd[:, ii, jj - 1] - nd[:, ii + 1, jj - 1]
            if smooth_curvatures is not None:
                for idx in range(nd.shape[0]):
                    nd[idx, :, jj] = np.convolve(
                        nd[idx, :, jj], smooth_curvatures_kernel, mode="same"
                    )

        nd = nd[:, :, 1:]
        evs = np.zeros((s_points_T.shape[0], s_points_T.shape[1], max_order + 1))

        for ii in range(s_points_T.shape[1]):
            cur_dim_vals = nd[:, ii, :]
            Q, _ = np.linalg.qr(cur_dim_vals)
            evs[:, ii, :] = Q[:, : max_order + 1]

        gc = np.zeros((max_order, s_points_T.shape[1]))

        for jj in range(max_order):
            for ii in range(s_points_T.shape[1] - 1):
                gc[jj, ii] = np.abs(
                    np.dot(evs[:, ii, jj] - evs[:, ii + 1, jj], evs[:, ii, jj + 1])
                ) / np.linalg.norm(nd[:, ii, 0])

        return gc

    def belongs(self, point, atol=gs.atol):
        """Test whether a point belongs to the curve.

        Test that all points of the curve belong to the ambient manifold.

        Parameters
        ----------
        point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Point representing a discrete curve.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : bool
            Boolean evaluating if point belongs to the space of discrete
            curves.
        """

        def each_belongs(pt):
            return gs.all(self.ambient_manifold.belongs(pt))

        if isinstance(point, list) or point.ndim > 2:
            return gs.stack([each_belongs(pt) for pt in point])

        return each_belongs(point)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at a curve.

        A vector is tangent at a curve if it is a vector field along that
        curve.

        Parameters
        ----------
        vector : array-like, shape=[..., n_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        ambient_manifold = self.ambient_manifold
        shape = vector.shape
        stacked_vec = gs.reshape(vector, (-1, shape[-1]))
        stacked_point = gs.reshape(base_point, (-1, shape[-1]))
        is_tangent = ambient_manifold.is_tangent(stacked_vec, stacked_point, atol)
        is_tangent = gs.reshape(is_tangent, shape[:-1])
        return gs.all(is_tangent, axis=-1)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        As tangent vectors are vector fields along a curve, each component of
        the vector is projected to the tangent space of the corresponding
        point of the discrete curve. The number of sampling points should
        match in the vector and the base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n_sampling_points, ambient_dim]
            Vector.
        base_point : array-like, shape=[..., n_sampling_points, ambient_dim]
            Discrete curve.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n_sampling_points, ambient_dim]
            Tangent vector at base point.
        """
        ambient_manifold = self.ambient_manifold
        shape = vector.shape
        stacked_vec = gs.reshape(vector, (-1, shape[-1]))
        stacked_point = gs.reshape(base_point, (-1, shape[-1]))
        tangent_vec = ambient_manifold.to_tangent(stacked_vec, stacked_point)
        tangent_vec = gs.reshape(tangent_vec, vector.shape)
        return tangent_vec

    def random_point(self, n_samples=1, bound=1.0, n_sampling_points=10):
        """Sample random curves.

        If the ambient manifold is compact, a uniform distribution is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Bound of the interval in which to sample for non compact
            ambient manifolds.
            Optional, default: 1.
        n_sampling_points : int
            Number of sampling points for the discrete curves.
            Optional, default : 10.

        Returns
        -------
        samples : array-like, shape=[..., n_sampling_points, {dim, [n, n]}]
            Points sampled on the hypersphere.
        """
        sample = self.ambient_manifold.random_point(n_samples * n_sampling_points)
        sample = gs.reshape(sample, (n_samples, n_sampling_points, -1))
        return sample[0] if n_samples == 1 else sample
