import numpy as np
import scipy.sparse.linalg as spsla
import skfem
from skfem.assembly import Basis
from skfem.helpers import inner, dot, grad

from scipy.stats._multivariate import multi_rv_frozen
from scipy._lib._util import check_random_state


class bilaplacian:
    def __init__(
        self,
        V: Basis,
        gamma: float,
        delta: float,
        Theta: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        robin_bc: bool = False,
        seed=None,
    ):
        r"""
        Create an infinite-dimensional Gaussian measure with bi-Laplacian covariance
        operator. That is, covariance given by the operator $C = (\delta I + \gamma
        {\rm div} \Theta \nabla)^{-2}$.

        Parameters
        ----------
        V: Basis
            Finite element discritization of the space
        gamma: float
            Covariance parameter
        delta: float
            Covariance parameter
        Theta: np.ndarray | None
            SPD tensor controlling anistrophic diffusion.
        mean : ArrayLike, default: ``0``
            Mean of the distribution.
        robin_bc: bool
            Whether to employ a Robin boundary condition to minimize boundary artifacts.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        """
        self._V = V
        self._mean = mean if mean is not None else V.zeros()
        self._random_state = check_random_state(seed)

        self.M = skfem.BilinearForm(lambda u, v, _: dot(u, v)).assemble(V)
        self.Minv = spsla.factorized(self.M)
        self.sqrtM = self.M.copy()
        self.sqrtM.setdiag(1 / np.sqrt(self.M @ V.ones()))
        self.sqrtM = self.M @ self.sqrtM

        @skfem.BilinearForm
        def bilaplacian_varf(trial, test, data):
            if Theta is None:
                varf = gamma * inner(grad(trial), grad(test))
            else:
                varf = gamma * inner(Theta * grad(trial), grad(test))
            varf += delta * inner(trial, test)
            return varf

        @skfem.BilinearForm
        def robin(trial, test, data):
            return (gamma * np.sqrt(delta / gamma) / 1.42) * inner(trial, test)

        self.A = bilaplacian_varf.assemble(V)
        if robin_bc:
            self.A += robin.assemble(V.boundary())
        self.Ainv = spsla.factorized(self.A)

    def logpdf(self, x) -> np.ndarray:
        pass

    def pdf(self, x) -> float:
        pass

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Get ``size`` random samples from the underlying distribution.
        """
        pass

    @property
    def V(self) -> Basis:
        """
        Get the underyling basis object for the measure.
        """
        return self._V

    @property
    def mean(self) -> np.ndarray:
        """
        Get the underyling basis object for the measure.
        """
        return self._mean

    @property
    def random_state(self):
        """Get or set the Generator object for generating random variates.

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)
