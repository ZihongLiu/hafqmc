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


def _parse_nelec(nelec: Union[int, Sequence[int]]) -> Tuple[int, Optional[Tuple[int, int]]]:
    if nelec is None:
        raise ValueError("Number of electrons must be provided for lattice models")
    if isinstance(nelec, (tuple, list)):
        if len(nelec) == 2:
            nup, ndn = (int(ne) for ne in nelec)
            return nup + ndn, (nup, ndn)
        if len(nelec) == 1:
            total = int(nelec[0])
            return total, None
    elif isinstance(nelec, (int, onp.integer)):
        return int(nelec), None
    raise ValueError(f"Unsupported electron configuration: {nelec}")


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
    nelec: int
    onsite_u: float
    mu: float = 0.0
    periodic: bool = True
    spin_counts: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.nelec <= 0:
            raise ValueError("nelec must be positive")
        self.lattice = RectangularLattice(self.dims, periodic=self.periodic)
        self.nbasis = 2 * self.lattice.n_sites
        if self.nelec > self.nbasis:
            raise ValueError("Number of electrons exceeds available orbitals")

    def _basis_index(self, site: int, spin: int) -> int:
        # spin: 0 for up block, 1 for down block
        return site + spin * self.lattice.n_sites

    def build_one_body(self) -> NumpyArray:
        h1e = onp.zeros((self.nbasis, self.nbasis), dtype=onp.float64)
        for site in range(self.lattice.n_sites):
            neighbors = {
                "x": self.lattice.neighbor(site, axis=0, direction=1),
                "y": self.lattice.neighbor(site, axis=1, direction=1),
            }
            for axis, neighbor in neighbors.items():
                if neighbor is None:
                    continue
                for spin, (t_x, t_y) in enumerate(((self.tx_up, self.ty_up),
                                                   (self.tx_dn, self.ty_dn))):
                    hop = -t_x if axis == "x" else -t_y
                    i = self._basis_index(site, spin)
                    j = self._basis_index(neighbor, spin)
                    h1e[i, j] = h1e[j, i] = hop
        if self.mu:
            h1e -= self.mu * onp.eye(self.nbasis)
        if self.onsite_u:
            nsite = self.lattice.n_sites
            h1e[:nsite, :nsite] -= self.onsite_u * onp.eye(nsite)
        return h1e

    def build_charge_cholesky(self) -> NumpyArray:
        diag_mask = onp.zeros((self.lattice.n_sites, self.nbasis), dtype=onp.float64)
        for site in range(self.lattice.n_sites):
            diag_mask[site, self._basis_index(site, 0)] = 1.0
            diag_mask[site, self._basis_index(site, 1)] = 1.0
        coeff = onp.sqrt(self.onsite_u)
        eye = onp.eye(self.nbasis, dtype=onp.float64)
        return coeff * diag_mask[..., None] * eye

    def build_reference_wfn(self, h1e: Optional[NumpyArray] = None) -> NumpyArray:
        if h1e is None:
            h1e = self.build_one_body()
        if self.spin_counts is None:
            eigvals, eigvecs = onp.linalg.eigh(h1e)
            order = onp.argsort(eigvals)
            return eigvecs[:, order[:self.nelec]]
        n_up, n_dn = self.spin_counts
        n_site = self.lattice.n_sites
        if n_up + n_dn != self.nelec:
            raise ValueError("spin_counts do not sum to total number of electrons")
        wfn = onp.zeros((self.nbasis, self.nelec), dtype=h1e.dtype)
        eval_up, vec_up = onp.linalg.eigh(h1e[:n_site, :n_site])
        eval_dn, vec_dn = onp.linalg.eigh(h1e[n_site:, n_site:])
        idx_up = onp.argsort(eval_up)[:n_up]
        idx_dn = onp.argsort(eval_dn)[:n_dn]
        wfn[:n_site, :n_up] = vec_up[:, idx_up]
        wfn[n_site:, n_up:n_up+n_dn] = vec_dn[:, idx_dn]
        return wfn

    def build_hamiltonian(self) -> Hamiltonian:
        h1e = self.build_one_body()
        ceri = self.build_charge_cholesky()
        wfn0 = self.build_reference_wfn(h1e)
        aux = {
            "lattice": {
                "dims": self.dims,
                "tx_up": self.tx_up,
                "ty_up": self.ty_up,
                "tx_dn": self.tx_dn,
                "ty_dn": self.ty_dn,
                "mu": self.mu,
                "spin_counts": self.spin_counts,
            },
            "type": "charge_hubbard_2d",
            "lattice_hubbard": {
                "U": float(self.onsite_u),
                "nsite": int(self.lattice.n_sites),
            },
        }
        return Hamiltonian(
            h1e=jnp.asarray(h1e),
            ceri=jnp.asarray(ceri),
            enuc=0.0,
            wfn0=jnp.asarray(wfn0),
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
    nelec_total, spin_counts = _parse_nelec(nelec_val)
    onsite_u = _cfg_value(interaction_cfg, "U")
    if onsite_u is None:
        raise ValueError("On-site interaction strength `U` must be specified")
    periodic = bool(_cfg_value(lattice_cfg, "periodic", True))
    model = ChargeChannelHubbard2D(
        dims=dims,
        tx_up=float(tx_up),
        ty_up=float(ty_up),
        tx_dn=float(tx_dn),
        ty_dn=float(ty_dn),
        nelec=int(nelec_total),
        onsite_u=float(onsite_u),
        mu=float(mu),
        periodic=periodic,
        spin_counts=spin_counts,
    )
    return model.build_hamiltonian()
