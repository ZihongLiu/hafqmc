# Hybrid AFQMC

This is the repo of corresponding code for the manuscript:
> Chen, Y., Zhang, L., E, W. & Car, R. (2022). Hybrid Auxiliary Field Quantum Monte Carlo for Molecular Systems. arXiv preprint [arXiv:2211.10824](https://arxiv.org/pdf/2211.10824.pdf).

You will need to use python and install `jax`, `flax`, `optax`, `pyscf`, `ml-collections` and `tensorboardX` to run the code. 

The [`hafqmc`](./hafqmc/) folder contains all the code and can be used directly as a package. Just make sure to add it to your `PYTHONPATH`. 

The [`examples`](./examples/) folder contains several examples that is shown in the manuscript. They can be directly run by something like `python run.py`. Note the `run_fprestart.py` will require you to run a optimization first and rename the `checkpoint.pkl` to `oldstates.pkl`.

## Lattice models

The package now includes a bare-bones lattice builder (`hafqmc/lattice.py`) that can construct a 2D Hubbard model with a charge-channel Hubbard–Stratonovich decomposition and anisotropic hopping that mixes spin-up and spin-down sectors (treated as a single large matrix). To use it, supply a `lattice` block together with the usual `hamiltonian` settings in your config before calling `train.train(cfg)`, e.g.

```python
cfg.lattice = {
    "dims": (4, 4),
    "t": 1.0,
    "alpha": 0.8,    # ty,up = alpha * tx,up and tx,dn = ty,up
    "nelec": (8, 6), # (n_up, n_dn) also supported
    "periodic": True,
}
cfg.hamiltonian = {"U": 4.0}
```

When `nelec` is a tuple `(n_up, n_dn)` the reference determinant fills each spin block separately, and the one-body Hamiltonian automatically includes the requested `-U Σ_i n_{i,↑}` shift on the up-spin block before applying the charge-channel HS decomposition.

When `cfg.lattice` is provided, training automatically builds the lattice Hamiltonian (otherwise it falls back to molecular or UEG systems as before).
