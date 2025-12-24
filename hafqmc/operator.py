import jax
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence, Union
from functools import partial

from .utils import _t_real, _t_cplx, ExpmFnType
from .utils import fix_init, symmetrize, Serial, cmult, scatter
from .utils import warp_spin_expm, make_expm_apply
from .hamiltonian import _align_rdm, calc_rdm


class OneBody(nn.Module):
    init_hmf: jnp.ndarray
    parametrize: bool = False
    init_random: float = 0.
    hermite_out: bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_hmf.shape[-1]

    def setup(self):
        if self.parametrize:
            self.hmf = self.param("hmf", fix_init, 
                self.init_hmf, self.dtype, self.init_random)
        else:
            self.hmf = self.init_hmf

    def __call__(self, step):
        hmf = symmetrize(self.hmf) if self.hermite_out else self.hmf
        hmf = cmult(step, hmf)
        return hmf
    
    @property
    def expm_apply(self):
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return warp_spin_expm(make_expm_apply(*_expm_op))


class AuxField(nn.Module):
    init_vhs: jnp.ndarray
    v_const : jnp.ndarray
    sq_alpha1 : float
    sq_alpha2 : float
    parametrize: bool = False
    init_random: float = 0.
    hermite_out: bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_vhs.shape[-1]

    @property
    def nfield(self):
        return self.init_vhs.shape[0]

    def setup(self):
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, 
                self.init_vhs, self.dtype, self.init_random)
        else:
            self.vhs = self.init_vhs

    def __call__(self, step, fields):
        vhs = self.vhs
        v_alpha = 1j * step * self.sq_alpha1 * \
                jnp.sum(fields[0,:]*vhs*self.v_const)
        log_weight = -0.5 * jnp.sum(jnp.square(fields)) + v_alpha
        
        real_field = step * self.sq_alpha2 * fields[1, :]
        imag_field = step * self.sq_alpha1 * fields[0, :]
        
        cmplx_field_up =  real_field + 1j * imag_field
        cmplx_field_dn = -real_field + 1j * imag_field

        vhs_sum_up = cmplx_field_up * vhs
        vhs_sum_dn = cmplx_field_dn * vhs

        return vhs_sum_up, vhs_sum_dn, log_weight
    
    @property
    def expm_apply(self):
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op

        def op_hub_exp_rmult(A,B):
            expA = jnp.exp(A)
            nB = expA[:,None]*B
            return nB

        return op_hub_exp_rmult
