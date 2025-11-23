import sys
sys.path.append("../..")

from ml_collections import config_dict

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
    cfg.hamiltonian = config_dict.ConfigDict({"U": 4.0})

    prop = cfg.ansatz.propagators[0]
    prop.max_nhs = 64
    prop.aux_network = None
    prop.init_tsteps = [0.05] * 3
    prop.sqrt_tsvpar = True
    prop.init_random = 0.05
    prop.mf_subtract = False

    cfg.optim.iteration = 2000
    cfg.optim.lr.start = 0.0
    cfg.optim.lr.decay = 1.0
    cfg.optim.optimizer = "adam"
    cfg.optim.sr = {
        "damping": 1e-3,
        "step_size": 0.5,
        "maxiter": 10,
        "tol": 1e-6,
    }

    cfg.sample.size = 1024
    cfg.sample.batch = 128
    cfg.sample.sampler = {"name": "metropolis", "sigma": 0.05, "steps": 10}
    cfg.sample.burn_in = 100

    cfg.loss.sign_factor = 0.0
    cfg.loss.std_factor = 0.0

    cfg.seed = 0
    cfg.log.level = "INFO"

    return cfg


if __name__ == "__main__":
    train.train(get_config())

