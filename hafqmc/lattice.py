import numpy as np
import scipy.linalg

def get_hubbard_square_lattice(nx, ny, t=1.0, periodic=True):
    """
    Generates the hopping matrix (h1e) for a 2D square Hubbard model.

    Args:
        nx (int): Number of sites in the x-direction.
        ny (int): Number of sites in the y-direction.
        t (float): Hopping parameter.
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
            h1e[i, j] = h1e[j, i] = -t
        elif periodic and nx > 1:
            j = (x + 1 - nx) + y * nx
            h1e[i, j] = h1e[j, i] = -t

        # Hopping in y-direction
        if y + 1 < ny:
            j = x + (y + 1) * nx
            h1e[i, j] = h1e[j, i] = -t
        elif periodic and ny > 1:
            j = x + (y + 1 - ny) * nx
            h1e[i, j] = h1e[j, i] = -t
            
    return h1e

def get_hubbard_eri_chol(nsites, U):
    """
    Generates the Cholesky-decomposed ERI for the Hubbard model.
    This is a "fake" Cholesky decomposition that represents the on-site U term.

    Args:
        nsites (int): Total number of lattice sites.
        U (float): The on-site interaction strength.

    Returns:
        numpy.ndarray: The Cholesky vectors `ceri` of shape (nsites, nsites, nsites).
    """
    if U == 0:
        return np.zeros((0, nsites, nsites))
    
    # For U > 0, we use the spin-2 HS decomposition which acts on (n_i - 1)^2
    # This requires a field coupling to sqrt(U/2)*(n_i-1).
    # However, the current code uses a charge decomposition for molecules.
    # Let's stick to a simpler representation that fits the existing code.
    # We can represent U * n_i_up * n_i_down with a single cholesky vector per site.
    ceri = np.zeros((nsites, nsites, nsites))
    if U > 0:
        val = np.sqrt(U)
        for i in range(nsites):
            ceri[i, i, i] = val
    else:
        # For U < 0 (attractive Hubbard), the HS transformation is different.
        # It couples to (n_i_up - n_i_down). This requires a more complex change.
        # For now, we focus on U > 0. A simple trick is to use imaginary fields.
        raise NotImplementedError("Attractive Hubbard model (U < 0) requires code changes to the propagator.")

    return ceri

def get_hubbard_wfn0(h1e, n_elec):
    """
    Gets the initial wavefunction (wfn0) from the non-interacting ground state.

    Args:
        h1e (numpy.ndarray): The hopping matrix.
        n_elec (tuple): A tuple of (n_up, n_down) electrons.

    Returns:
        tuple: A tuple of (wfn_alpha, wfn_beta) numpy arrays.
    """
    if h1e.ndim != 2:
        raise ValueError("h1e must be a 2D matrix.")
    
    n_up, n_down = n_elec
    eigvals, eigvecs = scipy.linalg.eigh(h1e)
    
    # Fill up the orbitals for alpha and beta electrons
    wfn_alpha = eigvecs[:, :n_up]
    wfn_beta = eigvecs[:, :n_down]
    
    return (wfn_alpha, wfn_beta)

def hamiltonian_from_lattice(lattice_cfg, elec_cfg):
    """
    A factory function to create a Hamiltonian tuple from lattice configurations.
    
    Args:
        lattice_cfg (dict): Dictionary with lattice parameters, e.g.,
                            {'name': 'hubbard', 'nx': 4, 'ny': 4, 't': 1.0, 'U': 4.0}
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
    
    t = lattice_cfg.get("t", 1.0)
    U = lattice_cfg.get("U", 4.0)
    periodic = lattice_cfg.get("periodic", True)

    n_up = elec_cfg.get("n_up", nsites // 2)
    n_down = elec_cfg.get("n_down", nsites // 2)

    # 1. Build h1e
    h1e = get_hubbard_square_lattice(nx, ny, t, periodic)
    
    # 2. Build ceri
    ceri = get_hubbard_eri_chol(nsites, U)
    
    # 3. Nuclear repulsion energy is 0 for a simple lattice model
    enuc = 0.0
    
    # 4. Build initial wavefunction
    wfn0 = get_hubbard_wfn0(h1e, (n_up, n_down))
    
    print(f"# Built Hubbard model: {nx}x{ny}, t={t}, U={U}, n_elec=({n_up},{n_down})")
    
    return h1e, ceri, enuc, wfn0