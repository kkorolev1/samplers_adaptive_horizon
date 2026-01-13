"""
Code for Trajectory Balance (TB) training with Importance Weighting (IW).
For further details see: https://arxiv.org/abs/2301.12594 and https://arxiv.org/abs/2501.06148
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp
import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.dds.dds_rnd import cos_sq_fn_step_scheme
from algorithms.gfn_tb.gfn_tb_rnd import rnd, loss_fn
from algorithms.gfn_tb.utils import get_invtemp
from eval.utils import extract_last_entry
from utils.print_utils import print_results


def gfn_tb_iw_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)

    dim = target.dim
    alg_cfg = cfg.algorithm
    batch_size = alg_cfg.batch_size
    num_steps = alg_cfg.num_steps
    reference_process = alg_cfg.reference_process
    noise_schedule = alg_cfg.noise_schedule
    loss_type = alg_cfg.loss_type

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Define initial and target density
    if reference_process in ["pinned_brownian", "ou"]:
        initial_dist = None
        aux_tuple = (dim, noise_schedule)
    elif reference_process == "ou_dds":
        initial_dist = distrax.MultivariateNormalDiag(
            jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
        )
        alphas = cos_sq_fn_step_scheme(num_steps, noise_scale=alg_cfg.noise_scale)
        alpha_fn = lambda step: alphas[step]
        aux_tuple = (alg_cfg.init_std, alpha_fn)
    else:
        raise ValueError(f"Reference process {reference_process} not supported.")

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    rnd_partial_base = partial(
        rnd,
        reference_process=reference_process,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=num_steps,
        use_lp=alg_cfg.model.use_lp,
        initial_dist=initial_dist,
    )
    loss_fn_base = partial(loss_fn, loss_type=loss_type, logr_clip=alg_cfg.logr_clip)

    # Define the function to be JIT-ed for FWD pass
    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params, invtemp=1.0):
        # prior_to_target=True, terminal_xs=None
        rnd_p = partial(rnd_partial_base, batch_size=batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

    # Define the function to be JIT-ed for FWD pass
    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def iw_loss_fwd_grad_fn(key, model_state, params, invtemp=1.0):
        # prior_to_target=True, terminal_xs=None
        rnd_p = partial(rnd_partial_base, batch_size=batch_size, prior_to_target=True)
        return loss_fn_base(
            key,
            model_state,
            params,
            rnd_p,
            invtemp=invtemp,
            importance_weighting=True,
            target_ess=alg_cfg.target_ess,
        )

    ### Prepare eval function
    eval_fn, logger = get_eval_fn(
        partial(rnd_partial_base, batch_size=cfg.eval_samples), target, target_xs, cfg
    )
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    # ### Initialise logZ with approximated logZ from a forward pass
    # @partial(jax.jit)
    # def loss_fwd_nograd_fn(key, model_state, params, invtemp=1.0):
    #     rnd_p = partial(rnd_partial_base, batch_size=batch_size, prior_to_target=True)
    #     return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

    # key, key_gen = jax.random.split(key_gen)
    # _, (_, log_pbs_over_pfs, log_rewards, _) = loss_fwd_nograd_fn(
    #     key, model_state, model_state.params
    # )
    # logZ_init += jax.nn.logsumexp(log_pbs_over_pfs + log_rewards) - jnp.log(batch_size)

    # model_state.params["params"]["logZ"] = jnp.atleast_1d(logZ_init)
    # print(f"logZ_init: {logZ_init:.4f}")

    ### Training phase
    for it in range(alg_cfg.iters):
        invtemp = get_invtemp(
            it, alg_cfg.iters // 2, alg_cfg.init_invtemp, (alg_cfg.init_invtemp < 1.0)
        )

        # On-policy training without importance weighting
        if it % (alg_cfg.iw_train_freq + 1) == 0:
            # Sample from model
            key, key_gen = jax.random.split(key_gen)
            grads, (xs, log_pbs_over_pfs, log_rewards, losses) = loss_fwd_grad_fn(
                key, model_state, model_state.params, invtemp=invtemp
            )
            model_state = model_state.apply_gradients(grads=grads)

        # Off-policy training with importance weighting
        else:
            # Sample from model
            key, key_gen = jax.random.split(key_gen)
            grads, (xs, log_pbs_over_pfs, log_rewards, losses) = iw_loss_fwd_grad_fn(
                key, model_state, model_state.params, invtemp=invtemp
            )
            model_state = model_state.apply_gradients(grads=grads)

        if cfg.use_wandb:
            wandb.log({"tb_loss": jnp.mean(losses)}, step=it)
            if loss_type == "tb":
                wandb.log({"logZ_learned": model_state.params["params"]["logZ"]}, step=it)

        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * batch_size)  # FIXME

            logger.update(eval_fn(model_state, key))

            print_results(it, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger), step=it)
