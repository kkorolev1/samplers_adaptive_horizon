"""
Code for Trajectory Balance (TB) training.
For further details see: https://arxiv.org/abs/2301.12594 and https://arxiv.org/abs/2501.06148
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp

import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.gfn_non_acyclic.buffer import build_terminal_state_buffer
from algorithms.gfn_non_acyclic.gfn_non_acyclic_rnd import rnd, rnd_eval, loss_fn
from algorithms.gfn_non_acyclic.utils import get_invtemp
from eval.utils import extract_last_entry
from utils.print_utils import print_results


def gfn_non_acyclic_ula(cfg, target, exp=None):
    key_gen = jax.random.PRNGKey(cfg.seed)

    dim = target.dim
    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size
    num_steps = alg_cfg.num_steps

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    initial_dist = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    aux_tuple = alg_cfg.step_size

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    rnd_eval_partial_base = partial(
        rnd_eval,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=num_steps,
        initial_dist=initial_dist,
        reference_process="ula",
    )

    ### Prepare eval function
    eval_fn, logger = get_eval_fn(
        partial(rnd_eval_partial_base, batch_size=cfg.eval_samples),
        target,
        target_xs,
        cfg,
    )
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    #         logZ_estimates.append(jax.nn.logsumexp(log_pbs_over_pfs + log_rewards))
    #     logZ_init = jax.nn.logsumexp(jnp.stack(logZ_estimates)) - jnp.log(
    #         buffer_cfg.prefill_steps * batch_size
    #     )
    # else:
    #     key, key_gen = jax.random.split(key_gen)
    #     _, (_, log_pbs_over_pfs, log_rewards, _) = loss_fwd_nograd_fn(
    #         key, model_state, model_state.params
    #     )
    #     logZ_init = jax.nn.logsumexp(log_pbs_over_pfs + log_rewards) - jnp.log(batch_size)

    # model_state.params["params"]["logZ"] = jnp.atleast_1d(logZ_init)
    # print(f"logZ_init: {logZ_init:.4f}")

    ### Training phase
    for it in range(alg_cfg.iters):
        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * batch_size)  # FIXME
            logger.update(eval_fn(model_state, key))
            print_results(it, logger, cfg)
            # if cfg.use_wandb:
            #     wandb.log(extract_last_entry(logger), step=it)
            if cfg.use_cometml:
                last_entry = extract_last_entry(logger)
                metrics = {}
                for key, value in last_entry.items():
                    if isinstance(value, wandb.Image):
                        exp.log_image(value.image, name=key, step=it)
                    else:
                        metrics[key] = value
                exp.log_metrics(metrics, step=it)
        break
