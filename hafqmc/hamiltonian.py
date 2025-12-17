import jax
import numpy as onp
from jax import numpy as jnp
from jax import scipy as jsp

from .utils import tree_map

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

def calc_pot_hubbard(rdm, nao, ham_u):
    gd, gl = _align_rdm(rdm, nao)
    # gl (2,nao,nao)
    if gl.ndim == 3 and gl.shape[-1] == nao:
        gl_0 = gl[0,:,:]
        gl_1 = gl[1,:,:]
        return ham_u * jnp.einsum("ii,ii", gl_0, gl_1)
    # gl (2,2,nao,nao)
    if gl.ndim == 4 and gl.shape[-1] == nao:
        gl_00 = gl[0,0,:,:]
        gl_10 = gl[1,0,:,:]
        gl_01 = gl[0,1,:,:]
        gl_11 = gl[1,1,:,:]
        return ham_u * ( jnp.einsum("ii,ii", gl_00, gl_11) - jnp.einsum("ii,ii", gl_01, gl_10) )
    raise ValueError("unknown gl type in calc_pot_Hubbard")
   
    
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


def calc_kin(h1e, rdm):
    # jnp.einsum("ij,ij", h1e, rdm)
    gd, gl = _align_rdm(rdm, h1e.shape[-1])
    return (h1e * gd).sum()


def calc_kin_hubbard(h1e, rdm):
    """One-body plus Hartree shift for lattice Hubbard models."""
    e_kin = (h1e * rdm).sum()
    return e_kin

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

class Hamiltonian:

    def __init__(self, h1e, v_hub, v_const, wfn0, aux=None, *, full_eri=False):
        self.h1e = jnp.asarray(h1e)
        self.v_hub = jnp.asarray(v_hub)
        self.v_const = jnp.asarray(v_const)
        self.wfn0 = tree_map(jnp.asarray, wfn0)
        self.aux = aux if aux is not None else {}
        self.nbasis = self.h1e.shape[-1]
        lattice_meta = self.aux.get("lattice_hubbard")
        
        lattice_meta = dict(lattice_meta)
        lattice_meta.setdefault("nsite", self.nbasis // 2)
        self._lattice_hubbard = lattice_meta

    def calc_kin(self, rdm):
        #return calc_kin(self.h1e, rdm)
        return calc_kin_hubbard(self.h1e, rdm)
    
    def calc_pot(self, rdm):
        return calc_pot_hubbard(rdm, self._lattice_hubbard["nsite"],
                                    self._lattice_hubbard["U"])
    
    calc_ovlp  = staticmethod(calc_ovlp)
    calc_slov  = staticmethod(calc_slov)
    calc_rdm   = staticmethod(calc_rdm)
    calc_theta = staticmethod(calc_theta)

    def local_energy(self, bra=None, ket=None, optimize=True):
        """the normalized energy from two slater determinants"""
        bra = bra if bra is not None else self.wfn0
        ket = ket if ket is not None else self.wfn0
        le_fn = self.local_energy
        return le_fn(bra, ket)

    def local_energy(self, bra, ket):
        rdm = calc_rdm(bra, ket)
        return self.calc_kin(rdm) + self.calc_pot(rdm)
    
    #def local_energy(self, bra, ket):
    #    bra, ket = _align_wfn(bra, ket)
    #    theta = calc_theta(bra, ket)
    #    rdm = calc_rdm_opt(bra, theta)
    #    return self.calc_kin(rdm) + self.calc_pot(rdm)

    def make_proj_op(self):
        """generate the modified hmf, vhs for projection"""
        return self.h1e, self.v_hub, self.v_const

    def to_tuple(self):
        return (self.h1e, self.v_hub, self.v_const, self.wfn0, self.aux)
