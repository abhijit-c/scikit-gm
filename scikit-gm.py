import numpy as np
import scipy as sp
import scipy.sparse.linalg as spsla
import skfem
from skfem.assembly import Basis
from skfem.helpers import inner, dot, grad

from scipy._lib._util import check_random_state

from typing import Literal


class Bilaplacian:
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

    def trace(
        self,
        method: Literal["exact", "estimator", "randomized"] = "exact",
        **kwargs,
    ) -> float:
        r"""
        Evaluate or estimate the trace of the covariance operator.

        Parameters
        ----------
        method: Literal["exact", "estimator", "randomized"], default: ``"exact"``
            Method to use to compute/estimate the trace. "exact" computes the diagonal
            and sums elements, "estimator" uses a Hutchinson estimator, "randomized"
            uses a low-rank appproximation.
        kwargs:
            Extra arguments to pass to the estimator. If the method is "estimator", see
            the ``tr_hutch`` function. If the method is "randomized", see the
            ``random_ghep`` function.
        """
        RinvM = spsla.LinearOperator(
            dtype=np.float64,
            shape=self.M.shape,
            matvec=lambda x: self.Rinv @ (self.M @ x),
        )
        if method == "exact":
            RinvM_I = RinvM @ sp.sparse.eye(self.V.N)
            return np.sum(RinvM_I.diagonal())

        if "seed" in kwargs:
            seed = kwargs.pop("seed")
        else:
            seed = self.random_state

        if method == "estimator":
            return tr_hutch(RinvM, seed=seed, **kwargs)
        elif method == "randomized":
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Method not recognized! Recieved {method=}")

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
        r"""
        Evaluate the gradient "logpdf" of the distribution.

        Note: An infinite dimensional distribution does not admit a pdf in this manner.
        However, this is simply a notational convinience to represent a similar
        computation.
        """
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


def tr_hutch(A, k=128, dist="rademacher", seed=None) -> float:
    rng = check_random_state(seed)
    A = spsla.aslinearoperator(A)
    n, m = A.shape
    if dist == "rademacher":
        r = rng.binomial(1, 0.5, size=(n, k))
        Omega = np.ones_like(r)
        Omega[r == 0] = -1
    elif dist == "gaussian":
        Omega = rng.standard_normal((n, k))
    else:
        raise ValueError("dist not recognized! Must be 'rademacher' or 'gaussian'.")
    return np.average(np.einsum("ij,ij->j", Omega, A @ Omega))


def random_ghep(A, B, Binv, k=128, p=20, twopass=True, seed=None):
    r"""
    Randomized algorithm for Generalized Hermitian Eigenvalue problem
    $A approx (BU) * \Lambda *(BU)^*$.

    Computes $k$ largest eigenvalues and eigenvectors.

    Modified from randomized algorithm for EV/SVD of $A$.

    This code is modified from the repository ``https://github.com/arvindks/kle``

    Parameters
    ----------
    A: np.ndarray | sp.sparray | LinearOperator
        $A$ as in $A \approx (BU) \Lambda (BU)^*$.
    B: np.ndarray | sp.sparray | LinearOperator
        $B$ as in $A \approx (BU) \Lambda (BU)^*$.
    Binv: np.ndarray | sp.sparray | LinearOperator
        Operator to solve the system $Bu = r$.
    k: int
        Number of eigenpairs to compute
    p: int, default: ``20``
        Oversampling parameter which can improve accuracy of resulting solution
    twopass: bool, default: ``True``
        Whether to use a twopass or onepass algorithm. Twopass is more accurate, but
        onepass is faster.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance
        then that instance is used.
    """
    A = spsla.aslinearoperator(A)
    B = spsla.aslinearoperator(B)
    Binv = spsla.aslinearoperator(Binv)

    m, n = A.shape
    if m != n:
        raise ValueError(f"A must be a square matrix! Recieved {A.shape=}")

    # Oversample
    k = k + p

    # Initialize quantities
    rng = check_random_state(seed)
    Omega = rng.standard_normal((n, k))
    Y = np.zeros_like(Omega)
    Yh = np.zeros_like(Omega)

    # Form matrix vector products with C = B^{-1}A
    for i in range(k):
        Y[:, i] = Binv @ (A @ Omega[:, i])

    # Compute Y = Q*R such that Q'*B*Q = I, R can be discarded
    q, Bq, _ = mgs_stable(B, Y)

    T = np.zeros((k, k))

    if twopass:
        for i in np.arange(k):
            Aq = A @ q[:, i]
            for j in np.arange(k):
                T[i, j] = np.dot(Aq, q[:, j])
    else:
        for i in np.arange(k):
            Yh[:, i] = B @ Y[:, i]

        OAO = np.dot(Omega.T, Yh)
        QtBO = np.dot(Bq.T, Omega)
        T = np.dot(sp.linalg.inv(QtBO.T), np.dot(OAO, sp.linalg.inv(QtBO)))

    # Eigen subproblem
    w, v = np.linalg.eigh(T)

    # Reverse eigenvalues in descending order
    w = w[::-1]

    # Compute eigenvectors
    u = np.dot(q, v[:, ::-1])
    k = k - p

    return w[:k], u[:, :k]


def mgs_stable(A, Z, verbose=False):
    """
        Returns QR decomposition of Z. Q and R satisfy the following relations
        in exact arithmetic

        1. QR    	= Z
        2. Q^*AQ 	= I
        3. Q^*AZ	= R
        4. ZR^{-1}	= Q

        Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
        for computing the A-orthogonal QR factorization

    This code is modified from the repository ``https://github.com/arvindks/kle``

        Parameters
        ----------
        A : {sparse matrix, dense matrix, LinearOperator}
                An array, sparse matrix, or LinearOperator representing
                the operation ``A * x``, where A is a real or complex square matrix.

        Z : ndarray

        verbose : bool, optional
                  Displays information about the accuracy of the resulting QR
                  Default: False

        Returns
        -------

        q : ndarray
                The A-orthogonal vectors

        Aq : ndarray
                The A^{-1}-orthogonal vectors

        r : ndarray
                The r of the QR decomposition


        See Also
        --------
        mgs : Modified Gram-Schmidt without re-orthogonalization
        precholqr  : Based on CholQR


        References
        ----------
        .. [1] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized
                Hermitian Eigenvalue Problems with application to computing
                Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885

        .. [2] W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980

        Examples
        --------

        >>> import numpy as np
        >>> A = np.diag(np.arange(1,101))
        >>> Z = np.random.randn(100,10)
        >>> q, Aq, r = mgs_stable(A, Z, verbose = True)

    """

    # Get sizes
    m = np.size(Z, 0)
    n = np.size(Z, 1)

    # Convert into linear operator
    Aop = spsla.aslinearoperator(A)

    # Initialize
    Aq = np.zeros_like(Z, dtype="d")
    q = np.zeros_like(Z, dtype="d")
    r = np.zeros((n, n), dtype="d")

    reorth = np.zeros((n,), dtype="d")
    eps = np.finfo(np.float64).eps

    q = np.copy(Z)

    for k in np.arange(n):
        Aq[:, k] = Aop.matvec(q[:, k])
        t = np.sqrt(np.dot(q[:, k].T, Aq[:, k]))

        nach = 1
        u = 0
        while nach:
            u += 1
            for i in np.arange(k):
                s = np.dot(Aq[:, i].T, q[:, k])
                r[i, k] += s
                q[:, k] -= s * q[:, i]

            Aq[:, k] = Aop.matvec(q[:, k])
            tt = np.sqrt(np.dot(q[:, k].T, Aq[:, k]))
            if tt > t * 10.0 * eps and tt < t / 10.0:
                nach = 1
                t = tt
            else:
                nach = 0
                if tt < 10.0 * eps * t:
                    tt = 0.0

        reorth[k] = u
        r[k, k] = tt
        tt = 1.0 / tt if np.abs(tt * eps) > 0.0 else 0.0
        q[:, k] *= tt
        Aq[:, k] *= tt

    return q, Aq, r
