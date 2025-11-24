import sys
#sys.path.append("../..")
#sys.path.append( "/home/zhliu/Seafile/mycode/hafqmc/" )
sys.path.append( "/home/bingxing2/home/scx6a1w/hafqmc_sr" )

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
    cfg.hamiltonian = config_dict.ConfigDict({"U": 4.0})

    prop = cfg.ansatz.propagators[0]
    prop.max_nhs = 1000
    prop.aux_network = None
    prop.init_tsteps = [0.01]
    prop.sqrt_tsvpar = True
    prop.init_random = 0.1
    prop.mf_subtract = False

    cfg.optim.iteration = 100
    cfg.optim.lr.start = 0.0
    cfg.optim.lr.decay = 1.0
    cfg.optim.optimizer = "adam"
    cfg.optim.sr = {
        "damping": 1e-3,
        "step_size": 6,
        "maxiter": 5000,
        "tol": 1e-7,
    }
    cfg.optim.collect_logder = True

    cfg.sample.batch = 2000
    #cfg.sample.sampler = {"name": "metropolis", "sigma": 0.05, "steps": 10}
    cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.0}
    cfg.sample.burn_in = 100

    cfg.loss.sign_factor = 0.0
    cfg.loss.std_factor = 0.0

    cfg.seed = 0
    cfg.log.level = "info"

    return cfg


if __name__ == "__main__":
    train.train(get_config())
