import numpy as np
def lse(A, b, B, d, cond=None):
    """
    Equality-contrained least squares.The following algorithm minimizes
    ||Ax - b|| subject to the constrain Bx = d.

    Parameters
    ----------
    A : array-like, shape=[m, n]
    b : array-like, shape=[m]
    B : array-like, shape=[p, n]
    d : array-like, shape=[p]
    cond : float, optional Cutoff for 'small' singular
    values; used to determine effective rank of A. Singular values smaller
    than rcond largest singular value are considered zero.

    Reference
    ---------
    Matrix Computations, Golub & van Loan, algorithm 12.1.2

    Examples
    --------
    >>> A, b = [[0, 2, 3], [1, 3, 4.5]], [1, 1]
    >>> B, d = [[1, 1, 0]], [1]
    >>> lse(A, b, B, d) array([-0.5 , 1.5 , -0.66666667])
    """
    from scipy import linalg
    if not hasattr(linalg, 'solve_triangular'): # compatibility for old scipy
        def solve_triangular(X, y, **kwargs):
            return linalg.solve(X, y)
        else:
            solve_triangular = linalg.solve_triangular
    A, b, B, d = map(np.asanyarray, (A, b, B, d))
    p = B.shape[0]
    Q, R = linalg.qr(B.T)
    y = solve_triangular(R[:p, :p], d, trans='T', lower=False)
    A = np.dot(A, Q)
    z = linalg.lstsq(A[:, p:], b - np.dot(A[:, :p], y), cond=cond)[0].ravel()
    return np.dot(Q[:, :p], y) + np.dot(Q[:, p:], z)