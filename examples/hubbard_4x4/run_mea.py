import os
import sys
# Add the parent directory to the python path so we can import hafqmc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml_collections import ConfigDict
from hafqmc.train import train

def get_config():
    cfg = ConfigDict()

    # System parameters for a 4x4 Hubbard model at half-filling
    cfg.lattice = {
        'name': 'hubbard',
        'nx': 4,
        'ny': 4,
        't_up_x': 1.0,
        't_up_y': 0.1,
        'U': -4.0,
        'periodic': True
    }
    cfg.electrons = {
        'n_up': 6,
        'n_down': 6
    }

    # Ansatz configuration
    cfg.ansatz = {
        'name': 'default',
        'init_tsteps': [0.1]*4, # Initial time step for the propagator
        'parametrize': 'wfn,tsteps,hmf',  # Parametrize the wavefunction directly
        'init_random': 1e-2,
        'use_complex': False
    }
    cfg.trial = None # Use a single Slater determinant ansatz, no multi-determinant trial

    # Sampler configuration
    cfg.sample = {
        'sampler': 'mala',
        'size': 2560, # Number of walkers
        'batch': 512,
        'prop_steps': 20, # Propagation steps per block
        'burn_in': 100
    }

    # Optimizer configuration
    cfg.optim = {
        'optimizer': {'name': 'adam'},
        'lr': {'start': 0, 'decay': None, 'delay': 0},
        'iteration': 150,
        'batch': None, # Use sample_batch for evaluation
        'grad_clip': 1.0,
        'baseline': {'decay': 0.98}
    }

    # Loss function configuration
    cfg.loss = {
        'sign_factor': 2,
        'sign_target': 0.6,
        'sign_power': 2,
        'std_factor': 0.00
    }

    # Logging and checkpointing
    cfg.log = {
        'stat_path': 'stats.json',
        'ckpt_path': 'checkpoint.pkl',
        'hamil_path': 'hamiltonian.pkl',
        'hpar_path': 'hparams.yaml',
        'stat_freq': 1,
        'ckpt_freq': 100,
        'level': 'info'
    }

    # Restart configuration (if any)
    cfg.restart = {
        'hamiltonian': 'hamiltonian.pkl',
        'params': 'checkpoint.pkl',
        'states': None
    }
    
    cfg.seed = 42

    return cfg

if __name__ == '__main__':
    config = get_config()
    train(config)
