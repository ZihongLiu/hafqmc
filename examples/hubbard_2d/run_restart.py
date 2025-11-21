import sys
from pathlib import Path

sys.path.append( "/home/bingxing2/home/scx6a1w/hafqmc" )

from hafqmc import train
from run_optimize import get_config as get_base_config

def get_restart_config():
    cfg = get_base_config()
    ckpt_path = Path(cfg.log.ckpt_path)
    hamil_path = Path(cfg.log.hamil_path)
    cfg.restart.hamiltonian = str(hamil_path)
    cfg.restart.states = str(ckpt_path)
    cfg.restart.params = None
    cfg.sample.burn_in = 0
    return cfg


if __name__ == "__main__":
    train.train(get_restart_config())
