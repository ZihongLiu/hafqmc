import numpy as onp
import jax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.tree_util import tree_map, tree_reduce

from .utils import PyTree

def _collapse_score_batch(score: PyTree, ref: PyTree):
    def _collapse(s, r):
        extra = s.ndim - r.ndim
        if extra <= 0:
            return s.reshape((1,) + s.shape)
        lead = int(onp.prod(s.shape[:extra]))
        return s.reshape((lead,) + s.shape[extra:])
    return tree_map(_collapse, score, ref)


def _center_score(score: PyTree):
    mean = tree_map(lambda s: jnp.mean(s, axis=0, keepdims=True), score)
    return tree_map(lambda s, m: s - m, score, mean)


def fisher_diag(score: PyTree):
    """Mean square of score along sample axis."""
    return tree_map(lambda s: jnp.mean(jnp.abs(s) ** 2, axis=0), score)

def fisher_diag_keepdims(score: PyTree):
    """Mean square of score along sample axis."""
    return tree_map(lambda s: jnp.mean(jnp.abs(s) ** 2, axis=0, keepdims=True), score)


def fisher_vector_product(score: PyTree, vec: PyTree, damping: float = 0.0):
    """Compute (F + damping I) v using score samples."""
    inner = tree_reduce(
        lambda x, y: x + y,
        tree_map(lambda s, v: jnp.einsum('i..., ...->i', jnp.conj(s), v), score, vec))
    def _prod(s):
        return jnp.einsum('i,i...->...', inner, s) / s.shape[0]
    fvp = tree_map(_prod, score)
    if damping and damping > 0:
        fvp = tree_map(lambda f, v: f + damping * v, fvp, vec)
    return fvp


def cg_solve(matvec, b: PyTree, maxiter: int = 5000, tol: float = 1e-6):
    """Solve matvec(x)=b with jax.scipy.sparse.linalg.cg on a PyTree."""
    sol_flat, info = jsp.sparse.linalg.cg(matvec, b, tol=tol, maxiter=maxiter)
    return sol_flat, info
