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
        "alpha": 0.1,
        "nelec": (6, 10),
        "periodic": True,
    })
    #cfg.molecule = config_dict.ConfigDict()  # unused when lattice is provided
    cfg.hamiltonian = config_dict.ConfigDict({"U": 4.0})

    cfg.ansatz.wfn_spinmix = False
    
    cfg.ansatz.propagators[0].max_nhs = 100
    cfg.ansatz.propagators[0].aux_network = None
    cfg.ansatz.propagators[0].init_tsteps = [0.01]
    cfg.ansatz.propagators[0].sqrt_tsvpar = True
    cfg.ansatz.propagators[0].init_random = 0.01
    cfg.ansatz.propagators[0].hermite_ops = False
    cfg.ansatz.propagators[0].mf_subtract = False
    cfg.ansatz.propagators[0].spin_mixing = False

    cfg.optim.optimizer = "adabelief"
    cfg.optim.grad_clip = 1.0
    cfg.optim.iteration = 1000
    cfg.optim.lr.start = 3e-4
    cfg.optim.lr.delay = 5000
    cfg.optim.lr.decay = 1

    cfg.sample.batch = 1000
    cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.0}
    cfg.sample.burn_in = 100

    cfg.loss.sign_factor = 999.0
    cfg.loss.std_factor = 0.7

    cfg.seed = 1
    cfg.log.level = "info"

    return cfg

if __name__ == "__main__":
    train.train(get_config())
