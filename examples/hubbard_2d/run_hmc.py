import os
import sys

from ml_collections import config_dict

# Allow running from the examples directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hafqmc import train
from hafqmc import config as hconfig

def get_config():
    cfg = hconfig.example()

    cfg.lattice = config_dict.ConfigDict({
        "dims": (4, 4),
        "t": 1.0,
        "alpha": 1.0,
        "nelec": (8, 8),
        "periodic": True,
    })
    cfg.hamiltonian = config_dict.ConfigDict({"U": 4.0, "alpha_u": 0.3,})

    # Keep propagator simple and stable for a tiny test.
    cfg.ansatz.propagators[0].init_tsteps = [0.10]
    cfg.ansatz.propagators[0].sqrt_tsvpar = True
    cfg.ansatz.propagators[0].ortho_intvl = 5
    cfg.ansatz.propagators[0].init_random = 0.0
    cfg.ansatz.propagators[0].hermite_ops = True
    cfg.ansatz.propagators[0].spin_mixing = False
    
    #cfg.ansatz.propagators[0].parametrize = 'tsteps,wfn,hmf'
    cfg.ansatz.propagators[0].parametrize = 'none'
    
    # Natural gradient with a plain update to exercise Fisher/CG.
    #cfg.optim.optimizer = {"name": "natural_grad", "base": "adam"}
    #cfg.optim.natgrad = {
    #    "damping": 1e-2,
    #    "approx": "cg",
    #    "cg": {"maxiter": 50000, "tol": 1e-6},
    #    "precond": None,
    #    "rescale_damping": 0,
    #    "update_mode": "plain",  # apply natgrad directly, no extra optimizer precond
    #}
    cfg.optim.grad_clip = 1.0
    cfg.optim.iteration = 200
    cfg.optim.lr.start = 0.0
    cfg.optim.lr.delay = 1e3
    cfg.optim.lr.decay = 0.9

    cfg.sample.size = 200
    cfg.sample.batch = 100
    #cfg.sample.sampler = {"name": "mcmc", "sigma": 0.05, "steps": 5}
    cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.0}
    #cfg.sample.sampler = {"name": "mala"}
    cfg.sample.burn_in = 1

    cfg.loss.sign_factor = 999
    cfg.loss.std_factor = 0.0

    cfg.seed = 42
    cfg.log.level = "info"
    cfg.log.stat_path = "tbdata_fisher/"
    cfg.log.ckpt_path = "checkpoint_fisher.pkl"

    return cfg

if __name__ == "__main__":
    train.train(get_config())
