import sys
from pathlib import Path

#sys.path.append("../..")
sys.path.append( "/home/bingxing2/home/scx6a1w/hafqmc_bond" )

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
    cfg.hamiltonian = config_dict.ConfigDict({
        "U": 4.0,
        "bond_J": 0.5,
    })

    cfg.ansatz.wfn_spinmix = False
    prop = cfg.ansatz.propagators[0]
    prop.max_nhs = 100
    prop.aux_network = None
    prop.init_tsteps = [0.01]
    prop.sqrt_tsvpar = True
    prop.init_random = 0.01
    prop.hermite_ops = False
    prop.mf_subtract = False
    prop.spin_mixing = False
    prop.extra_vhs = "bond_ops"

    cfg.optim.optimizer = "adabelief"
    cfg.optim.grad_clip = 1.0
    cfg.optim.iteration = 3000
    #cfg.optim.lr.start = 3e-4
    #cfg.optim.lr.delay = 5000
    #cfg.optim.lr.decay = 1.0
    cfg.optim.lr.start = 1e-3
    cfg.optim.lr.delay = 3000
    cfg.optim.lr.decay = 0.9

    cfg.sample.batch = 2000
    cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.0}
    cfg.sample.burn_in = 100

    cfg.loss.sign_factor = 999.0
    cfg.loss.std_factor = 0.7

    cfg.seed = 17
    cfg.log.level = "info"

    return cfg

if __name__ == "__main__":
    train.train(get_config())
