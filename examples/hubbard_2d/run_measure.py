import sys
from pathlib import Path

sys.path.append( "/home/bingxing2/home/scx6a1w/hafqmc" )

from hafqmc import train
from run_optimize import get_config as get_base_config


def get_measure_config():
    cfg = get_base_config()
    ckpt_path = Path(cfg.log.ckpt_path)
    hamil_path = Path(cfg.log.hamil_path)
    cfg.restart.hamiltonian = str(hamil_path)
    cfg.restart.states = str(ckpt_path)
    cfg.restart.params = None
    cfg.optim.lr.start = 0.0
    cfg.optim.iteration = 10
    cfg.sample.burn_in = 0
    cfg.log.stat_freq = 1
    return cfg


if __name__ == "__main__":
    train.train(get_measure_config())
