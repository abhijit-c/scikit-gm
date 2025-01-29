import numpy as np
import scipy as sp
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
        self.Ainv = spsla.LinearOperator(
            dtype=np.float64, shape=self.A.shape, matvec=spsla.factorized(self.A)
        )

        mass = skfem.BilinearForm(lambda u, v, _: dot(u, v))

        # lump mass matrix if possible
        if isinstance(V.mesh, skfem.MeshTri):
            self._V_lumped = skfem.CellBasis(
                V.mesh, V.elem, quadrature=(V.elem.doflocs.T, np.full(3, 1 / 6))
            )
            self.M = mass.assemble(self._V_lumped)
            self.Minv, self.sqrtM, self.sqrtMinv = (self.M.copy() for _ in range(3))
            self.Minv.setdiag(1 / self.M.diagonal())
            self.sqrtM.setdiag(np.sqrt(self.M.diagonal()))
            self.sqrtMinv.setdiag(1 / np.sqrt(self.M.diagonal()))
        else:  # Going to be slow
            self.M = mass.assemble(V)
            self.Minv = spsla.LinearOperator(
                dtype=np.float64, shape=self.M.shape, matvec=spsla.factorized(self.M)
            )
            self.sqrtM = sp.linalg.sqrtm(M.todense())
            lu, piv = sp.linalg.lu_factor(self.sqrtM)
            self.sqrtMinv = spsla.LinearOperator(
                dtype=np.float64,
                shape=self.M.shape,
                matvec=lambda x: sp.linalg.lu_solve((lu, piv), x),
            )

        def R(self, x: np.ndarray) -> np.ndarray:
            return self.A @ (self.Minv @ (self.A @ x))

        self.R = spsla.LinearOperator(dtype=np.float64, shape=self.M.shape, matvec=R)

        def Rinv(self, x: np.ndarray) -> np.ndarray:
            return self.Ainv @ (self.M @ (self.Ainv @ x))

        self.Rinv = spsla.LinearOperator(
            dtype=np.float64, shape=self.M.shape, matvec=Rinv
        )

    def trace(self, method="exact") -> float:
        r"""
        Evaluate or estimate the trace of the covariance operator.
        """
        raise NotImplementedError("TODO")

    def logpdf(self, x: np.ndarray) -> float:
        r"""
        Evaluate the "logpdf" of the distribution.

        Note: An infinite dimensional distribution does not admit a pdf in this manner.
        However, this is simply a notational convinience to represent a similar
        computation.
        """
        innov = x - self.mean
        return 0.5 * np.inner(innov, self.R @ innov)

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        innov = x - self.mean
        return self.R @ innov

    def sample(self) -> np.ndarray:
        r"""
        Get a random sample from the underlying distribution.
        """
        s = self.random_state.standard_normal(self.V.N)
        return self.mean + self.sqrtM @ (self.Ainv @ s)

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Get ``size`` random samples from the underlying distribution.
        """
        if size == 1:
            return self.sample()
        return np.array([self.sample() for _ in range(size)])

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
