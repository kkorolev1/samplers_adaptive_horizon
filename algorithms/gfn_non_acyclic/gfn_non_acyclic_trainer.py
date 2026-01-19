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
from algorithms.gfn_non_acyclic.gfn_non_acyclic_rnd import (
    rnd_no_term,
    rnd_with_term,
    rnd_cont,
    loss_fn,
)
from algorithms.gfn_non_acyclic.utils import get_invtemp
from eval.utils import extract_last_entry
from utils.print_utils import print_results


def gfn_non_acyclic_trainer(cfg, target, exp=None):
    key_gen = jax.random.PRNGKey(cfg.seed)

    dim = target.dim
    alg_cfg = cfg.algorithm
    buffer_cfg = alg_cfg.buffer
    batch_size = alg_cfg.batch_size
    num_steps = alg_cfg.num_steps
    reg_coef = alg_cfg.reg_coef

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    initial_dist = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )
    aux_tuple = (alg_cfg.model.gamma,)

    # Initialize the buffer
    use_buffer = buffer_cfg.use
    buffer = buffer_state = None
    if use_buffer:
        buffer = build_terminal_state_buffer(
            dim=dim,
            max_length=buffer_cfg.max_length_in_batches * batch_size,
            prioritize_by=buffer_cfg.prioritize_by,
            target_ess=buffer_cfg.target_ess,
            sampling_method=buffer_cfg.sampling_method,
            rank_k=buffer_cfg.rank_k,
        )
        buffer_state = buffer.init(dtype=target_xs.dtype, device=target_xs.device)

    # Initialize the model
    key, key_gen = jax.random.split(key_gen)
    model_state = init_model(key, dim, alg_cfg)

    rnd_partial_base = partial(
        # trajectories sampled without / with termination resp.
        rnd_no_term if alg_cfg.no_term else rnd_with_term,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=num_steps,
        initial_dist=initial_dist,
    )
    local_search_cfg = alg_cfg.local_search
    rnd_local_search_partial_base = partial(
        rnd_cont,
        batch_size=batch_size,
        aux_tuple=(local_search_cfg.gamma,),
        target=target,
        num_steps=local_search_cfg.num_steps,
        step_name="mala",
        initial_dist=initial_dist,
    )
    rnd_eval_partial_base = partial(
        rnd_with_term,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=alg_cfg.eval_max_steps,
        initial_dist=initial_dist,
    )
    loss_fn_base = partial(loss_fn, logr_clip=alg_cfg.logr_clip, reg_coef=reg_coef)

    # Define the function to be JIT-ed for FWD pass
    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params, invtemp=1.0):
        # prior_to_target=True, terminal_xs=None
        rnd_p = partial(rnd_partial_base, batch_size=batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

    # --- Define the function to be JIT-ed for BWD pass ---
    @partial(jax.jit)
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_bwd_grad_fn(
        key, model_state, params, terminal_xs, log_rewards, invtemp=1.0
    ):
        # prior_to_target=False, terminal_xs is now an argument
        rnd_p = partial(
            rnd_partial_base,
            batch_size=batch_size,
            prior_to_target=False,
            terminal_xs=terminal_xs,
            log_rewards=log_rewards,
        )
        return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

    # Define the function to be JIT-ed for FWD pass without gradients
    @partial(jax.jit)
    def loss_fwd_nograd_fn(key, model_state, params, invtemp=1.0):
        rnd_p = partial(rnd_partial_base, batch_size=batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_p, invtemp=invtemp)

    ### Prepare eval function
    eval_fn, logger = get_eval_fn(
        partial(rnd_eval_partial_base, batch_size=cfg.eval_samples),
        target,
        target_xs,
        cfg,
    )
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    ### Prefill phase; initialise logZ with mean of approximated logZs
    if use_buffer and buffer_cfg.prefill_steps > 0:
        # logZ_estimates = []
        for _ in range(buffer_cfg.prefill_steps):
            key, key_gen = jax.random.split(key_gen)
            _, (samples, log_iws, log_rewards, losses) = loss_fwd_nograd_fn(
                key, model_state, model_state.params
            )
            buffer_state = buffer.add(
                buffer_state,
                samples,
                log_iws,
                log_rewards,
                losses,
            )
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

    def tree_l2_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return 0.0
        return jnp.sqrt(sum([jnp.sum(jnp.square(jnp.asarray(x))) for x in leaves]))

    grads = None
    off_policy_iters = 0
    ### Training phase
    for it in range(alg_cfg.iters):
        invtemp = get_invtemp(
            it, alg_cfg.iters // 2, alg_cfg.init_invtemp, (alg_cfg.init_invtemp < 1.0)
        )

        # On-policy training with forward samples
        if not use_buffer or it % (buffer_cfg.bwd_to_fwd_ratio + 1) == 0:
            # Sample from model
            key, key_gen = jax.random.split(key_gen)
            grads, (samples, log_iws, log_rewards, losses) = loss_fwd_grad_fn(
                key, model_state, model_state.params, invtemp=invtemp
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Add samples to buffer
            if use_buffer:
                buffer_state = buffer.add(
                    buffer_state,
                    samples,
                    log_iws,
                    log_rewards,
                    losses,
                )

            # from jax.scipy.special import logsumexp
            # logZ_est = logsumexp(log_pbs_over_pfs + log_rewards) - jnp.log(batch_size)
            # jax.debug.print("logZ_est: {logZ_est}", logZ_est=logZ_est)

        # Off-policy training with buffer samples
        else:
            if local_search_cfg.use and off_policy_iters % local_search_cfg.cycle == 0:
                # jax.debug.print(f"{it}, {off_policy_iters}")
                key, key_gen = jax.random.split(key_gen)
                trajectories, _, _, terminal_costs, trajectories_length, *_ = (
                    rnd_local_search_partial_base(
                        key,
                        model_state,
                        model_state.params,
                    )
                )
                buffer_state = buffer.add(
                    buffer_state,
                    trajectories[
                        jnp.arange(trajectories.shape[0]), trajectories_length - 1
                    ],
                    jnp.zeros_like(terminal_costs),
                    -terminal_costs,  # log_rewards
                    jnp.zeros_like(terminal_costs),
                )

            # Sample terminal states from buffer
            key, key_gen = jax.random.split(key_gen)
            samples, log_rewards, indices = buffer.sample(buffer_state, key, batch_size)

            # Get grads with the off-policy samples
            key, key_gen = jax.random.split(key_gen)
            grads, (_, log_pbs_over_pfs, _, losses) = loss_bwd_grad_fn(
                key,
                model_state,
                model_state.params,
                samples,
                log_rewards,
                invtemp=invtemp,
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Update scores in buffer if needed
            if buffer_cfg.update_score:
                buffer_state = buffer.update_priority(
                    buffer_state,
                    indices,
                    (log_pbs_over_pfs.sum(-1) + log_rewards),
                    log_rewards,
                    losses,
                )
            off_policy_iters += 1

        if cfg.use_cometml:
            params_norm = tree_l2_norm(
                model_state.params.get("params", model_state.params)
            )
            grads_norm = (
                tree_l2_norm(grads.get("params", grads))
                if grads is not None
                else jnp.nan
            )
            exp.log_metrics(
                {
                    "loss": jnp.mean(losses),
                    "logZ_learned": model_state.params["params"]["logZ"],
                    "params_l2_norm": params_norm,
                    "grads_l2_norm": grads_norm,
                },
                step=it,
            )

        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * batch_size)  # FIXME

            logger.update(eval_fn(model_state, key))

            # Evaluate buffer samples
            if use_buffer:
                from eval import discrepancies

                buffer_xs, _, _ = buffer.sample(buffer_state, key, cfg.eval_samples)

                for d in cfg.discrepancies:
                    if logger.get(f"discrepancies/buffer_samples_{d}", None) is None:
                        logger[f"discrepancies/buffer_samples_{d}"] = []
                    logger[f"discrepancies/buffer_samples_{d}"].append(
                        getattr(discrepancies, f"compute_{d}")(
                            target_xs, buffer_xs, cfg
                        )
                        if target_xs is not None
                        else jnp.inf
                    )
                vis_dict = target.visualise(samples=buffer_xs)
                logger.update(
                    {f"{key}_buffer_samples": value for key, value in vis_dict.items()}
                )

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
