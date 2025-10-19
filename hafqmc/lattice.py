import numpy as np
import scipy.linalg

def get_hubbard_square_lattice(nx, ny, t_x=1.0, t_y=1.0, periodic=True):
    """
    Generates the hopping matrix (h1e) for a 2D square Hubbard model.

    Args:
        nx (int): Number of sites in the x-direction.
        ny (int): Number of sites in the y-direction.
        t_x (float): Hopping parameter in the x-direction.
        t_y (float): Hopping parameter in the y-direction.
        periodic (bool): Whether to use periodic boundary conditions.

    Returns:
        numpy.ndarray: The h1e matrix of shape (N_sites, N_sites).
    """
    nsites = nx * ny
    h1e = np.zeros((nsites, nsites))

    for i in range(nsites):
        x, y = i % nx, i // nx

        # Hopping in x-direction
        if x + 1 < nx:
            j = (x + 1) + y * nx
            h1e[i, j] = h1e[j, i] = -t_x
        elif periodic and nx > 1:
            j = (x + 1 - nx) + y * nx
            h1e[i, j] = h1e[j, i] = -t_x

        # Hopping in y-direction
        if y + 1 < ny:
            j = x + (y + 1) * nx
            h1e[i, j] = h1e[j, i] = -t_y
        elif periodic and ny > 1:
            j = x + (y + 1 - ny) * nx
            h1e[i, j] = h1e[j, i] = -t_y
            
    return h1e

def get_hubbard_eri_chol(nsites, U):
    """
    Generates the Cholesky-decomposed ERI for the attractive Hubbard model (U < 0).

    Args:
        nsites (int): Total number of lattice sites.
        U (float): The on-site interaction strength (must be negative).

    Returns:
        numpy.ndarray: The Cholesky vectors `ceri` of shape (nsites, nsites, nsites).
    """
    if U >= 0:
        raise ValueError("This code is now configured for the attractive Hubbard model, so U must be negative.")
    
    ceri = np.zeros((nsites, nsites, nsites))
    val = np.sqrt(-U)
    for i in range(nsites):
        ceri[i, i, i] = val

    return ceri

def hamiltonian_from_lattice(lattice_cfg, elec_cfg):
    """
    A factory function to create a Hamiltonian tuple from lattice configurations.
    
    Args:
        lattice_cfg (dict): Dictionary with lattice parameters, e.g.,
                            {'name': 'hubbard', 'nx': 4, 'ny': 4, 't_up_x': 1.0, 't_up_y': 1.0}
        elec_cfg (dict): Dictionary with electron numbers, e.g.,
                         {'n_up': 8, 'n_down': 8}

    Returns:
        tuple: A tuple (h1e, ceri, enuc, wfn0) compatible with the Hamiltonian class.
    """
    name = lattice_cfg.get("name", "hubbard").lower()
    if name != "hubbard":
        raise NotImplementedError(f"Lattice model '{name}' is not implemented.")

    nx = lattice_cfg.get("nx", 4)
    ny = lattice_cfg.get("ny", 1)
    nsites = nx * ny
    
    # Handle spin-dependent hopping
    t = lattice_cfg.get("t")
    t_up_x = lattice_cfg.get("t_up_x", t if t is not None else 1.0)
    t_up_y = lattice_cfg.get("t_up_y", t if t is not None else 1.0)
    t_down_x = lattice_cfg.get("t_down_x", t_up_y)
    t_down_y = lattice_cfg.get("t_down_y", t_up_x)

    U = lattice_cfg.get("U", 4.0)
    periodic = lattice_cfg.get("periodic", True)

    n_up = elec_cfg.get("n_up", nsites // 2)
    n_down = elec_cfg.get("n_down", nsites // 2)

    # 1. Build h1e for each spin channel
    h1e_up = get_hubbard_square_lattice(nx, ny, t_x=t_up_x, t_y=t_up_y, periodic=periodic)
    h1e_down = get_hubbard_square_lattice(nx, ny, t_x=t_down_x, t_y=t_down_y, periodic=periodic)
    h1e = np.stack([h1e_up, h1e_down])
    
    # 2. Build ceri
    ceri = get_hubbard_eri_chol(nsites, U)
    
    # 3. Nuclear repulsion energy is 0 for a simple lattice model
    enuc = 0.0
    
    # 4. Build initial wavefunction from spin-dependent h1e
    eigvals_up, eigvecs_up = scipy.linalg.eigh(h1e_up)
    wfn_alpha = eigvecs_up[:, :n_up]
    
    eigvals_down, eigvecs_down = scipy.linalg.eigh(h1e_down)
    wfn_beta = eigvecs_down[:, :n_down]

    wfn0 = (wfn_alpha, wfn_beta)
    
    print(f"# Built Hubbard model: {nx}x{ny}, U={U}, n_elec=({n_up},{n_down})")
    print(f"# Hopping params (up): t_x={t_up_x}, t_y={t_up_y}")
    print(f"# Hopping params (down): t_x={t_down_x}, t_y={t_down_y}")
    
    return h1e, ceri, enuc, wfn0