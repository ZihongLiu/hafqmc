import dataclasses
from typing import Sequence, Tuple, Union, Optional, Any

import numpy as onp
from jax import numpy as jnp

from .hamiltonian import Hamiltonian


Number = Union[int, float]
NumpyArray = onp.ndarray


def _cfg_value(cfg: Any, name: str, default=None):
    """Retrieve an attribute/key from a ConfigDict/dict/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    if hasattr(cfg, name):
        return getattr(cfg, name)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except TypeError:
            pass
    return default


def _parse_nelec(nelec: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if nelec is None:
        raise ValueError("Explicit spin-resolved electron counts (n_up, n_dn) are required")
    if isinstance(nelec, (tuple, list)) and len(nelec) == 2:
        return tuple(int(ne) for ne in nelec)  # type: ignore
    raise ValueError(f"Unsupported electron configuration for spin-separated lattice model: {nelec}")


@dataclasses.dataclass
class RectangularLattice:
    dims: Sequence[int]
    periodic: bool = True

    def __post_init__(self):
        if len(self.dims) != 2:
            raise ValueError("Only 2D lattices are supported")
        self.dims = tuple(int(d) for d in self.dims)
        self.n_sites = int(onp.prod(self.dims))
        self.coords = onp.stack(onp.unravel_index(onp.arange(self.n_sites), self.dims), axis=-1)

    def neighbor(self, site_index: int, axis: int, direction: int = 1) -> Optional[int]:
        coord = self.coords[site_index].copy()
        coord[axis] += direction
        if self.periodic:
            coord[axis] %= self.dims[axis]
        else:
            if coord[axis] < 0 or coord[axis] >= self.dims[axis]:
                return None
        return onp.ravel_multi_index(tuple(coord), self.dims)


@dataclasses.dataclass
class ChargeChannelHubbard2D:
    dims: Tuple[int, int]
    tx_up: float
    ty_up: float
    tx_dn: float
    ty_dn: float
    nelec: Tuple[int, int]
    onsite_u: float
    alpha_u: float
    mu: float = 0.0
    periodic: bool = True

    def __post_init__(self):
        self.n_up, self.n_dn = _parse_nelec(self.nelec)
        if self.n_up < 0 or self.n_dn < 0:
            raise ValueError("Electron counts must be non-negative")
        self.lattice = RectangularLattice(self.dims, periodic=self.periodic)
        self.nbasis_spin = self.lattice.n_sites
        self.nbasis_total = 2 * self.nbasis_spin
        self.nelec_total = self.n_up + self.n_dn
        if self.nelec_total > self.nbasis_total:
            raise ValueError("Number of electrons exceeds available orbitals")

    def build_one_body(self) -> NumpyArray:
        h_up = onp.zeros((self.nbasis_spin, self.nbasis_spin), dtype=onp.float64)
        h_dn = onp.zeros_like(h_up)
        for site in range(self.nbasis_spin):
            neighbors = {
                "x": self.lattice.neighbor(site, axis=0, direction=1),
                "y": self.lattice.neighbor(site, axis=1, direction=1),
            }
            for axis, neighbor in neighbors.items():
                if neighbor is None:
                    continue
                hop_up = -(self.tx_up if axis == "x" else self.ty_up)
                hop_dn = -(self.tx_dn if axis == "x" else self.ty_dn)
                h_up[site, neighbor] = h_up[neighbor, site] = hop_up
                h_dn[site, neighbor] = h_dn[neighbor, site] = hop_dn
        if self.mu:
            eye = self.mu * onp.eye(self.nbasis_spin)
            h_up -= eye
            h_dn -= eye
        if self.onsite_u:
            h_up -= self.onsite_u * onp.eye(self.nbasis_spin)
        return onp.stack((h_up, h_dn))

    def build_hubbard_HS(self) -> NumpyArray:
        couple_list = onp.zeros((self.nbasis_spin),dtype=onp.float64)
        for site in range(self.nbasis_spin):
            couple_list[site] = 1.0
        coeff = onp.sqrt(self.onsite_u)
        # return coupling and const term
        return coeff * couple_list, -couple_list

    def build_reference_wfn(self, h1e: Optional[NumpyArray] = None):
        if h1e is None:
            h1e = self.build_one_body()
        h_up, h_dn = h1e
        lam = 0.01
        r_up = onp.diag(1 - 2*(onp.arange(self.nbasis_spin) % 2))
        r_dn =-onp.diag(1 - 2*(onp.arange(self.nbasis_spin) % 2))
        h_up_afm = h_up + lam * r_up
        h_dn_afm = h_dn + lam * r_dn
        eval_up, vec_up = onp.linalg.eigh(h_up_afm)
        eval_dn, vec_dn = onp.linalg.eigh(h_dn_afm)
        idx_up = onp.argsort(eval_up)[: self.n_up]
        idx_dn = onp.argsort(eval_dn)[: self.n_dn]
        w_up = vec_up[:, idx_up]
        w_dn = vec_dn[:, idx_dn]
        w_up = jnp.asarray(w_up)
        w_dn = jnp.asarray(w_dn)
        return (w_up, w_dn)

    def build_hamiltonian(self) -> Hamiltonian:
        h1e   = self.build_one_body()
        v_hub, v_const = self.build_hubbard_HS()
        wfn0  = self.build_reference_wfn(h1e)

        sqrt_alpha_charge = float(onp.sqrt(    self.alpha_u))
        sqrt_alpha_spin   = float(onp.sqrt(1.0-self.alpha_u))

        aux = {
            "lattice": {
                "dims": self.dims,
                "tx_up": self.tx_up,
                "ty_up": self.ty_up,
                "tx_dn": self.tx_dn,
                "ty_dn": self.ty_dn,
                "mu": self.mu,
                "n_up": self.n_up,
                "n_dn": self.n_dn,
            },
            "type": "charge_hubbard_2d",
            "lattice_hubbard": {
                "U": float(self.onsite_u),
                "sqrt_alpha_charge": sqrt_alpha_charge,
                "sqrt_alpha_spin": sqrt_alpha_spin,
                "nsite": int(self.lattice.n_sites),
            },
        }
        return Hamiltonian(
            h1e=jnp.asarray(h1e),
            v_hub=jnp.asarray(v_hub),
            v_const=v_const,
            wfn0=wfn0,
            aux=aux,
        )


def build_lattice_hamiltonian(lattice_cfg, interaction_cfg=None) -> Hamiltonian:
    model_type = _cfg_value(lattice_cfg, "model",
                            _cfg_value(lattice_cfg, "type", "hubbard2d")).lower()
    if model_type not in ("hubbard2d", "hubbard"):
        raise ValueError(f"Unsupported lattice model: {model_type}")
    dims = tuple(_cfg_value(lattice_cfg, "dims"))
    if len(dims) != 2:
        raise ValueError("`dims` must specify a 2D lattice, e.g. (Lx, Ly)")
    base_t = _cfg_value(lattice_cfg, "t", None)
    tx_up = _cfg_value(lattice_cfg, "tx_up", base_t)
    if tx_up is None:
        raise ValueError("Base hopping amplitude `t` or `tx_up` must be provided")
    alpha = _cfg_value(lattice_cfg, "alpha", 1.0)
    ty_up = _cfg_value(lattice_cfg, "ty_up", alpha * tx_up)
    tx_dn = _cfg_value(lattice_cfg, "tx_dn", ty_up)
    ty_dn = _cfg_value(lattice_cfg, "ty_dn", tx_up)
    mu = _cfg_value(lattice_cfg, "mu", _cfg_value(interaction_cfg, "mu", 0.0))
    nelec_val = _cfg_value(lattice_cfg, "nelec")
    n_up, n_dn = _parse_nelec(nelec_val)
    onsite_u = _cfg_value(interaction_cfg, "U")
    alpha_u = _cfg_value(interaction_cfg, "alpha_u")
    if onsite_u is None:
        raise ValueError("On-site interaction strength `U` must be specified")
    periodic = bool(_cfg_value(lattice_cfg, "periodic", True))
    model = ChargeChannelHubbard2D(
        dims=dims,
        tx_up=float(tx_up),
        ty_up=float(ty_up),
        tx_dn=float(tx_dn),
        ty_dn=float(ty_dn),
        nelec=(int(n_up), int(n_dn)),
        onsite_u=float(onsite_u),
        alpha_u=float(alpha_u),
        mu=float(mu),
        periodic=periodic,
    )
    return model.build_hamiltonian()
