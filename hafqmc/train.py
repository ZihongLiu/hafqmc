import time
import logging
import jax 
import optax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from optax._src import alias as optax_alias
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter
from typing import NamedTuple

from .hamiltonian import Hamiltonian
from .latt_setup import build_lattice_hamiltonian
from .ansatz import Ansatz, BraKet
from .estimator import make_eval_total
from .sampler import make_sampler, make_multistep, make_batched, SamplerUnion
from .utils import ensure_mapping, save_pickle, load_pickle, Printer, cfg_to_yaml
from .utils import make_moving_avg, PyTree, tree_map

#SR related
from .fisher import fisher_vector_product, fisher_diag, cg_solve
from .fisher import _collapse_score_batch, _center_score

def lower_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(target - s, 0) ** power

def upper_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(s - target, 0) ** power


def make_optimizer(name, lr_schedule, grad_clip=None, natgrad_cfg=None, **kwargs):
    natcfg = None
    base_name = name
    if natgrad_cfg is not None:
        natcfg = {"damping": 1e-3, "approx": "cg",
                  "cg": {"maxiter": 10, "tol": 1e-6},
                  "rescale_damping": None,
                  "precond": False,
                  "update_mode": "base"}  # base | plain
        for k, v in natgrad_cfg.items():
            if k == "cg":
                natcfg["cg"].update(v)
            else:
                natcfg[k] = v
        natcfg["lr_fn"] = lr_schedule if callable(lr_schedule) else None
    if name.lower() in ("natural_grad", "natgrad"):
        base_name = kwargs.pop("base", "adam")
        if natcfg is None:
            natcfg = {"damping": 1e-3, "approx": "cg",
                      "cg": {"maxiter": 10, "tol": 1e-6},
                      "rescale_damping": None,
                      "precond": False,
                      "update_mode": "base"}
            natcfg["lr_fn"] = lr_schedule if callable(lr_schedule) else None
    opt_fn = getattr(optax_alias, base_name)
    opt = opt_fn(lr_schedule, **kwargs)
    if grad_clip is not None:
        opt = optax.chain(optax.clip(grad_clip), opt)
    return opt, natcfg


def make_lr_schedule(start=1e-4, decay=1., delay=1e4):
    if decay is None:
        return start
    return lambda t: start * jnp.power((1.0 / (1.0 + (t/delay))), decay)


def make_loss(expect_fn, 
              sign_factor=0., sign_target=1., sign_power=2.,
              std_factor=0., std_target=1., std_power=2):

    def loss(params, data, *extra, **kwargs):
        e_tot, aux = expect_fn(params, data, *extra, **kwargs)
        loss = e_tot
        if sign_factor > 0:
            exp_s = aux["exp_s"]
            loss += lower_penalty(exp_s, sign_factor, sign_target, sign_power)
        if std_factor > 0:
            std_es = aux["std_es"]
            loss += upper_penalty(std_es, std_factor, std_target, std_power)
        return loss, aux
         
    return loss


class TrainingState(NamedTuple):
    step: int
    params: PyTree
    mc_state: PyTree
    opt_state: PyTree
    est_state: PyTree = None


def _apply_natgrad(grads, score, cfg):
    damping = cfg.get("damping", 0.0)
    rescale_damp = cfg.get("rescale_damping", damping)
    approx = cfg.get("approx", "cg")

    o_vec_0 = _collapse_score_batch(score, grads)
    o_vec = _center_score(o_vec_0)

    if approx == "diag":
        diag = fisher_diag(o_vec)
        return tree_map(lambda g, d: g / (d + damping), grads, diag)
    cg_kw = cfg.get("cg", {})
    if cfg.get("precond", False):
        diag = fisher_diag(o_vec)
        sqrt_diag = tree_map(lambda d: jnp.sqrt(d + rescale_damp), diag)
        o_vec_tilde = tree_map(lambda s, sd: s / sd, o_vec, sqrt_diag)
        grads_pre = tree_map(lambda g, sd: g / sd, grads, sqrt_diag)
        mv = lambda v: fisher_vector_product(o_vec_tilde, v, damping=damping)
        nat_g_pre, info = cg_solve(mv, grads_pre,
            maxiter=cg_kw.get("maxiter", 500), tol=cg_kw.get("tol", 1e-6))
        nat_g = tree_map(lambda ng, sd: ng / sd, nat_g_pre, sqrt_diag)
    else:
        mv = lambda v: fisher_vector_product(o_vec, v, damping=damping)
        nat_g, info = cg_solve(mv, grads,
            maxiter=cg_kw.get("maxiter", 500), tol=cg_kw.get("tol", 1e-6))

    return nat_g


def make_training_step(loss_and_grad, mc_sampler, optimizer, accumulator=None, natcfg=None):
    is_union = isinstance(mc_sampler, SamplerUnion)

    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, opt_state, ebar = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state = sampler.refresh(mc_state, params)
        mc_state, data = sampler.sample(key, params, mc_state)
        (loss, aux), grads = loss_and_grad(params, data, ebar)
        aux = dict(aux)
        if natcfg is not None and getattr(sampler, "compute_score", None) is not None:
            score = sampler.compute_score(data, params)
            if score is not None:
                flat_score, _ = ravel_pytree(score)
                abs_score = jnp.abs(flat_score)
                aux["score_mean"] = jnp.mean(abs_score)
                aux["score_std"] = jnp.std(abs_score)
                grads = _apply_natgrad(grads, score, natcfg)
        grads = tree_map(jnp.conj, grads) # for complex parameters
        if natcfg is not None and natcfg.get("update_mode", "base") == "plain":
            lr_fn = natcfg.get("lr_fn", None)
            lr_val = 0.005
            if lr_fn is not None:
                lr_val = lr_fn(ii)
            params = tree_map(lambda p, g: p - lr_val * g, params, grads)
            updates = opt_state
        else:
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        if accumulator is not None: ebar = accumulator(ebar, aux["e_tot"], ii)
        new_state = TrainingState(ii+1, params, mc_state, opt_state, ebar)
        return new_state, (loss, aux)

    return step


def make_evaluation_step(expect_fn, mc_sampler):
    is_union = isinstance(mc_sampler, SamplerUnion)
    
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, *other = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state, data = sampler.sample(key, params, mc_state)
        e_tot, aux = expect_fn(params, data)
        new_state = TrainingState(ii+1, params, mc_state, *other)
        return new_state, (e_tot, aux)
    
    return step
        

def train(cfg: ConfigDict):
    # handle logging
    logging.basicConfig(force=True, format='# [%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger("train")
    log_level = getattr(logging, cfg.log.level.upper())
    logger.setLevel(log_level)
    print_fields = {"step": "", "loss": ".4f", "e_tot": ".4f", 
                    "exp_es": ".4f", "exp_s": ".4f", "lw_mean": ".3f", "lw_std": ".3f",
                    "score_mean": ".3e", "score_std": ".3e"}
    if cfg.loss.std_factor >= 0:
        print_fields.update({"std_es": ".4f", "std_s": ".4f"})
    print_fields["lr"] = ".1e"
    printer = Printer(print_fields, time_format=".4f")
    if cfg.log.hpar_path:
        with open(cfg.log.hpar_path, "w") as hpfile:
            print(cfg_to_yaml(cfg), file=hpfile)

    # get the constants
    total_iter = cfg.optim.iteration
    sample_size = cfg.sample.size
    sample_batch = cfg.sample.batch
    if sample_size % sample_batch != 0:
        logger.warning("Sample size not divisible by batch size, rounding up")
    sample_step = -(-sample_size // sample_batch)
    sample_size = sample_batch * sample_step
    sample_prop = cfg.sample.prop_steps
    eval_batch = cfg.optim.batch if cfg.optim.batch is not None else sample_batch
    if sample_size % eval_batch != 0:
        logger.warning("Eval batch size not dividing sample size, using sample batch size")
        eval_batch = sample_batch

    # set up the hamiltonian
    if cfg.restart.hamiltonian is None:
        lattice_cfg = cfg.lattice
        logger.info("Building lattice Hamiltonian")
        hamiltonian = build_lattice_hamiltonian(lattice_cfg, cfg.hamiltonian)
        print(f"# Non-interacting lattice energy: {hamiltonian.local_energy()}")
        save_pickle(cfg.log.hamil_path, hamiltonian.to_tuple())
    else:
        logger.info("Loading Hamiltonian from saved file")
        hamil_data = load_pickle(cfg.restart.hamiltonian)
        hamiltonian = Hamiltonian(*hamil_data)
        print(f"# HF energy from loaded: {hamiltonian.local_energy()}")

    # set up all other classes and functions
    logger.info("Setting up the training loop")
    ansatz = Ansatz.create(hamiltonian, **cfg.ansatz)
    braket = BraKet(ansatz)
    sampler_1s_1c = make_sampler(braket,**ensure_mapping(cfg.sample.sampler, default_key="name"))
    sampler_1s_nc = make_batched(sampler_1s_1c, sample_batch, concat=False)
    mc_sampler = make_multistep(sampler_1s_nc, sample_step, concat=True)
    lr_schedule = make_lr_schedule(**cfg.optim.lr)
    optimizer, natcfg = make_optimizer(natgrad_cfg=cfg.optim.natgrad,
        lr_schedule=lr_schedule, grad_clip=cfg.optim.grad_clip,
        **ensure_mapping(cfg.optim.optimizer, default_key="name"))
    expect_fn = make_eval_total(hamiltonian, braket, 
        default_batch=eval_batch, calc_stds=True)
    loss_fn = make_loss(expect_fn, **cfg.loss)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    moving_avg_fn = (make_moving_avg(**cfg.optim.baseline)
        if cfg.optim.baseline is not None else None)

    # the core training iteration, to be pmaped
    if cfg.optim.lr.start > 0:
        train_step = make_training_step(loss_and_grad, mc_sampler, optimizer, moving_avg_fn, natcfg)
    else:
        train_step = make_evaluation_step(expect_fn, mc_sampler)
    train_step = jax.jit(train_step, static_argnames="sample_flag")
    
    # set up all states
    if cfg.restart.states is None:
        logger.info("Initializing parameters and states")
        key = jax.random.PRNGKey(cfg.seed)
        key, pakey, mckey = jax.random.split(key, 3)
        fshape = braket.fields_shape()
        if cfg.restart.params is None:
            params = jax.jit(braket.init)(pakey, tree_map(jnp.zeros, fshape))
        else:
            logger.info("Loading parameters from saved file")
            params = load_pickle(cfg.restart.params)
            if isinstance(params, tuple): params = params[1]
            if isinstance(params, tuple): params = params[1]
        mc_state = mc_sampler.init(mckey, params)
        opt_state = optimizer.init(params)
        if cfg.sample.burn_in > 0:
            logger.info(f"Burning in the sampler for {cfg.sample.burn_in} steps")
            key, subkey = jax.random.split(key)
            mc_state = sampler_1s_nc.burn_in(subkey, params, mc_state, cfg.sample.burn_in)
        ebar = hamiltonian.local_energy() if cfg.optim.baseline is not None else None
        train_state = TrainingState(0, params, mc_state, opt_state, ebar)
    else:
        logger.info("Loading parameters and states from saved file")
        key, *rest = load_pickle(cfg.restart.states)
        rest = rest[0] if len(rest) == 1 else (0, *rest)
        if len(rest) < 5 and cfg.optim.baseline is not None:
            rest = (*rest, hamiltonian.local_energy())
        train_state = TrainingState(*rest)

    # the actual training iteration
    logger.info("Start training")
    printer.print_header(prefix="# ")
    for ii in range(total_iter + 1):
        printer.reset_timer()
        # choose sampler
        sflag = None
        if not (sample_prop is None or isinstance(sample_prop, int)):
            key, flagkey = jax.random.split(key)
            sflag = sample_prop[jax.random.choice(flagkey, len(sample_prop))]
        # core training step
        key, subkey = jax.random.split(key)
        train_state, (loss, aux) = train_step(subkey, train_state, sample_flag=sflag)
        # logging anc checkpointing
        if ii % cfg.log.stat_freq == 0:
            if sflag is not None: aux["nprop"] = sflag
            _lr = (lr_schedule(train_state.opt_state[-1][0].count) 
                if callable(lr_schedule) else lr_schedule)
            printer.print_fields({"step": ii, "loss": loss, **aux, "lr": _lr})
            writer.add_scalars("stat", {"loss": loss, **aux, "lr": _lr}, global_step=ii)
        if ii % cfg.log.ckpt_freq == 0:
            save_pickle(cfg.log.ckpt_path, (key, tuple(train_state)))
    writer.close()
    
    return train_state
