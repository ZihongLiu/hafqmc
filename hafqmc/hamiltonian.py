import jax
import numpy as onp
from jax import numpy as jnp
from jax import scipy as jsp

from .utils import tree_map, scatter

def _has_spin(wfn):
    return not (isinstance(wfn, (jnp.ndarray, onp.ndarray)) 
                and wfn.ndim == 2)

def _make_ghf(wfn):
    assert _has_spin(wfn)
    wa, wb = wfn
    return jsp.linalg.block_diag(wa, wb)

def _align_wfn(V, U):
    uhf_v, uhf_u = _has_spin(V), _has_spin(U)
    if uhf_v and not uhf_u: #U is ghf
        V = _make_ghf(V)
    if not uhf_v and uhf_u: #V is ghf
        U = _make_ghf(U)
    return V, U

def _align_rdm(rdm, nao):
    # return sum diag block and all components
    if rdm.ndim == 2 and rdm.shape[-1] == nao:
        # rhf case, assume rdm is from single wfn
        return rdm*2, jnp.stack([rdm, rdm])
    if rdm.ndim == 3 and rdm.shape[0] in (2,4) and rdm.shape[-1] == nao:
        # uhf case, no non-diag block (or aligned ghf case)
        return rdm[0]+rdm[-1], rdm
    if rdm.ndim == 2 and rdm.shape[-1] == nao * 2:
        # ghf case, return four blocks in uu,ud,du,dd order
        lrdm = rdm.reshape(2,nao,2,nao).swapaxes(1,2)
        return lrdm[0,0]+lrdm[1,1], lrdm
    raise ValueError("unknown rdm type")

def calc_e2b_hubbard(rdm, nao, U_icf):
    gd, gl = _align_rdm(rdm, nao)
    # gl (2,nao,nao)
    if gl.ndim == 3 and gl.shape[-1] == nao:
        gl_0 = gl[0,:,:]
        gl_1 = gl[1,:,:]
        return U_icf * jnp.einsum("ii,ii", gl_0, gl_1)
    # gl (2,2,nao,nao)
    if gl.ndim == 4 and gl.shape[-1] == nao:
        gl_00 = gl[0,0,:,:]
        gl_10 = gl[1,0,:,:]
        gl_01 = gl[0,1,:,:]
        gl_11 = gl[1,1,:,:]
        return U_icf * ( jnp.einsum("ii,ii", gl_00, gl_11) - jnp.einsum("ii,ii", gl_01, gl_10) )
    raise ValueError("unknown gl type in calc_e2b_Hubbard")
   
    
def calc_ovlp_ns(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U, no spin index
    """
    return jnp.linalg.det( V.conj().T @ U )

def calc_ovlp(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U with spin components

    Args:
        V, U (array or tuple of array):
            matrix representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 

    Returns:
        ovlp (float):
            overlap of the two Slater determinants
    """
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_ovlp_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return calc_ovlp_ns(Va, Ua) * calc_ovlp_ns(Vb, Ub)


def calc_slov_ns(V, U):
    r"""
    Sign and log of overlap of two SD V and U, no spin index
    """
    return jnp.linalg.slogdet( V.conj().T @ U )

def calc_slov(V, U):
    r"""
    Sign and log of overlap of two SD V and U with spin components

    Args:
        V, U (array or tuple of array):
            matrix representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 

    Returns:
        sign (float):
            sign of the overlap of two SD
        logdet (float):
            log of the absolute value of the overlap
    """
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_slov_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    sa, la = calc_slov_ns(Va, Ua)
    sb, lb = calc_slov_ns(Vb, Ub)
    return sa * sb, la + lb


def calc_rdm_ns(V, U):
    r"""
    One-particle reduced density matrix. No spin index.
    """
    V_h = V.conj().T
    inv_O = jnp.linalg.inv(V_h @ U)
    rdm = U @ inv_O @ V_h
    return rdm.T

def calc_rdm(V, U):
    r"""
    One-particle reduced density matrix, with both spin components in a tuple
    
    Calculate the one particle RDM from two (non-orthogonal) 
    Slater determinants (V for bra and U for ket) for each spin components.
    
    .. math::
        \langle \phi_V | c_i^{\dagger} c_j | \phi_U \rangle =
        [U (V^{\dagger}U)^{-1} V^{\dagger}]_{ji}
        
    :math:`U` stands for the matrix representation of Slater determinant :math:`|\psi_U\rangle`.
    
    Args:
        V, U (array or tuple of array):
            matrix representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 
        
    Returns:
        rdm (array):
            spin up and spin down one-particle reduced density matrix in computing basis
    """
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_rdm_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return jnp.stack((calc_rdm_ns(Va, Ua), calc_rdm_ns(Vb, Ub)), 0)


def calc_e1b(h1e, rdm):
    # jnp.einsum("ij,ij", h1e, rdm)
    gd, gl = _align_rdm(rdm, h1e.shape[-1])
    return (h1e * gd).sum()


def calc_e1b_hubbard(h1e, rdm):
    """One-body plus Hartree shift for lattice Hubbard models."""
    e_kin = (h1e * rdm).sum()
    return e_kin


def calc_e2b(eri, rdm):
    gs, ga = _align_rdm(rdm, eri.shape[-1])
    if eri.ndim == 4:
        return calc_ej_dense(eri, gs) - calc_ek_dense(eri, ga)
    elif eri.ndim == 3:
        return calc_ej_chol(eri, gs) - calc_ek_chol(eri, ga)
    else:
        raise RuntimeError(f"invalid shape of ERI: {eri.shape}")

def calc_ej_dense(eri, srdm):
    return 0.5 * jnp.einsum("prqs,pr,qs", eri, srdm, srdm)

def calc_ej_chol(ceri, srdm):
    chol_j = jnp.einsum("kpr,pr->k", ceri, srdm)
    return 0.5 * jnp.einsum("k,k", chol_j, chol_j)

def calc_ek_dense(eri, rdm):
    assert rdm.ndim in (3, 4)
    subs = "prqs,lps,lqr" if rdm.ndim == 3 else "prqs,abps,baqr"
    return 0.5 * jnp.einsum(subs, eri, rdm, rdm)

def calc_ek_chol(ceri, rdm):
    assert rdm.ndim in (3, 4)
    chol_k = jnp.einsum("kpr,...ps->k...rs", ceri, rdm)
    subs = "klrs,klsr" if rdm.ndim == 3 else "kabrs,kbasr"
    return 0.5 * jnp.einsum(subs, chol_k, chol_k)


def calc_v0(eri):
    if eri.ndim == 4:
        return calc_v0_dense(eri)
    elif eri.ndim == 3:
        return calc_v0_chol(eri)
    else:
        raise RuntimeError(f"invalid shape of ERI: {eri.shape}")

def calc_v0_dense(eri):
    return jnp.einsum("prrs->ps", eri)

def calc_v0_chol(ceri):
    return jnp.einsum("kpr,krs->ps", ceri, ceri)


def calc_theta_ns(V, U):
    V_h = V.conj().T
    inv_O = jnp.linalg.inv(V_h @ U)
    return U @ inv_O

def calc_theta(V, U):
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_theta_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return (calc_theta_ns(Va, Ua), calc_theta_ns(Vb, Ub))


def calc_rdm_opt(V, theta):
    V, theta = _align_wfn(V, theta)
    if not _has_spin(V) and not _has_spin(theta):
        return theta @ V.conj().T
    Va, Vb = V
    tha, thb = theta
    return jnp.stack((tha @ Va.conj().T, thb @ Vb.conj().T), 0)


def calc_e2b_opt(ceri, bra, theta):
    bra, theta = _align_wfn(bra, theta)
    if _has_spin(bra) and _has_spin(theta):
        ej, ek = calc_ejk_opt_u(ceri, bra, theta)
    else:
        if bra.shape[0] == ceri.shape[-1]:
            ej, ek = calc_ejk_opt_r(ceri, bra, theta)
        else:
            ej, ek = calc_ejk_opt_g(ceri, bra, theta)
    return ej - ek

def calc_ejk_opt_r(ceri, bra, theta):
    f = jnp.einsum("kpq,pi,qj->kij", ceri, bra.conj(), theta)
    ej = 2 * jnp.sum(f.trace(0, -1, -2) ** 2)
    ek = jnp.einsum("kij,kji", f, f)
    return ej, ek

def calc_ejk_opt_u(ceri, bra, theta):
    ej = ek = 0.
    fup = jnp.einsum("kpq,pi,qj->kij", ceri, bra[0].conj(), theta[0])
    cup = fup.trace(0, -1, -2)
    ek += 0.5 * jnp.einsum("kij,kji", fup, fup)
    del fup
    fdn = jnp.einsum("kpq,pi,qj->kij", ceri, bra[1].conj(), theta[1])
    cdn = fdn.trace(0, -1, -2)
    ek += 0.5 * jnp.einsum("kij,kji", fdn, fdn)
    ej = 0.5 * jnp.sum((cup + cdn)**2)
    return ej, ek

def calc_ejk_opt_g(ceri, bra, theta):
    nao = ceri.shape[-1]
    nele = bra.shape[-1]
    bra = bra.reshape(2, nao, nele)
    theta = theta.reshape(2, nao, nele)
    f = jnp.einsum("kpq,api,aqj->kij", ceri, bra.conj(), theta)
    ej = 0.5 * jnp.sum(f.trace(0, -1, -2) ** 2)
    ek = 0.5 * jnp.einsum("kij,kji", f, f)
    return ej, ek


class Hamiltonian:

    def __init__(self, h1e, ceri, enuc, wfn0, aux=None, *, full_eri=False):
        self.h1e = jnp.asarray(h1e)
        self.ceri = jnp.asarray(ceri)
        self._eri = jnp.einsum("kpr,kqs->prqs", ceri, ceri) if full_eri else None
        self.enuc = enuc
        self.wfn0 = tree_map(jnp.asarray, wfn0)
        self.aux = aux if aux is not None else {}
        self.nbasis = self.h1e.shape[-1]
        lattice_meta = self.aux.get("lattice_hubbard")
        if lattice_meta is not None:
            lattice_meta = dict(lattice_meta)
            lattice_meta.setdefault("nsite", self.nbasis // 2)
            self._lattice_hubbard = lattice_meta
        else:
            self._lattice_hubbard = None

    def calc_e1b(self, rdm):
        if self._lattice_hubbard is not None:
            return calc_e1b_hubbard(self.h1e, rdm)
        return calc_e1b(self.h1e, rdm)
    
    def calc_e2b(self, rdm):
        if self._lattice_hubbard is not None:
            return calc_e2b_hubbard(rdm, self._lattice_hubbard["nsite"],
                                    self._lattice_hubbard["U"])
        eri = self.ceri if self._eri is None else self._eri
        return calc_e2b(eri, rdm)
    
    def calc_e2b_opt(self, bra, theta):
        return calc_e2b_opt(self.ceri, bra, theta)

    calc_ovlp  = staticmethod(calc_ovlp)
    calc_slov  = staticmethod(calc_slov)
    calc_rdm   = staticmethod(calc_rdm)
    calc_theta = staticmethod(calc_theta)

    def local_energy(self, bra=None, ket=None, optimize=True):
        """the normalized energy from two slater determinants"""
        bra = bra if bra is not None else self.wfn0
        ket = ket if ket is not None else self.wfn0
        if self._lattice_hubbard is not None:
            le_fn = self.local_energy_raw
        else:
            le_fn = (self.local_energy_opt 
                if optimize and self._eri is None else self.local_energy_raw)
        return le_fn(bra, ket)

    def local_energy_raw(self, bra, ket):
        rdm = calc_rdm(bra, ket)
        return self.enuc + self.calc_e1b(rdm) + self.calc_e2b(rdm)
    
    def local_energy_opt(self, bra, ket):
        bra, ket = _align_wfn(bra, ket)
        theta = calc_theta(bra, ket)
        rdm = calc_rdm_opt(bra, theta)
        return self.enuc + self.calc_e1b(rdm) + self.calc_e2b_opt(bra, theta)

    def make_proj_op(self, trial):
        """generate the modified hmf, vhs and enuc for projection"""
        eri = self.ceri if self._eri is None else self._eri
        hmf_raw = self.h1e #- 0.5 * calc_v0(eri)
        vhs_raw = self.ceri # vhs is real here, will time 1j in propagator
        if trial is None:
            return hmf_raw, vhs_raw, self.enuc
        rdm_t = calc_rdm(trial, trial)
        if rdm_t.ndim == 3:
            rdm_t = rdm_t.sum(0)
        vbar = jnp.einsum("kpq,pq->k", vhs_raw, rdm_t)
        enuc = self.enuc - 0.5 * (vbar**2).sum()
        hmf = hmf_raw + jnp.einsum('kpq,k->pq', vhs_raw, vbar)
        vhs = vhs_raw - vbar.reshape(-1,1,1) * jnp.eye(vhs_raw.shape[-1]) / rdm_t.trace()
        return hmf, vhs, enuc

    def to_tuple(self):
        return (self.h1e, self.ceri, self.enuc, self.wfn0, self.aux)
