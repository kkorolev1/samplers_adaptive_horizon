"""
Code for Sub-Trajectory Balance (SubTB) training (Note: SubTB generalizes DB and TB).
For further details see: https://arxiv.org/abs/2209.12782 and https://arxiv.org/abs/2501.06148
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp
import wandb

from algorithms.common.diffusion_related.init_model import init_model
from algorithms.common.eval_methods.stochastic_oc_methods import get_eval_fn
from algorithms.dds.dds_rnd import cos_sq_fn_step_scheme
from algorithms.gfn_subtb.visualise import (
    visualise_intermediate_distribution,
    visualise_true_intermediate_distribution,
)
from algorithms.gfn_subtb.gfn_subtb_rnd import (
    get_beta_fn,
    loss_fn_joint,
    loss_fn_subtb,
    loss_fn_subtb_lambda,
    loss_fn_joint_lambda,
    rnd,
)
from algorithms.gfn_subtb_smc.gfn_subtb_smc_rnd import batch_simulate_fwd_subtrajectories
from algorithms.gfn_tb.buffer import build_terminal_state_buffer
from algorithms.gfn_tb.sampling_utils import get_sampling_func
from algorithms.gfn_tb.utils import get_invtemp
from eval.utils import extract_last_entry
from utils.print_utils import print_results


def get_subtb_lambda_coefs(lamda: float, num_steps: int):
    range_vals = jnp.arange(num_steps + 1)
    diff_matrix = range_vals - range_vals.reshape(-1, 1)
    B = jnp.log(lamda) * diff_matrix
    B = jnp.where(diff_matrix <= 0, -jnp.inf, B)
    log_total_lambda = jax.nn.logsumexp(B.reshape(-1), axis=0)
    coef = jnp.exp(B - log_total_lambda)
    return coef


def gfn_subtb_smc_trainer(cfg, target):
    key_gen = jax.random.PRNGKey(cfg.seed)

    dim = target.dim
    alg_cfg = cfg.algorithm
    smc_cfg = alg_cfg.smc
    mcmc_cfg = alg_cfg.mcmc
    buffer_cfg = alg_cfg.buffer
    batch_size = alg_cfg.batch_size
    num_steps = alg_cfg.num_steps
    n_chunks = alg_cfg.n_chunks
    reference_process = alg_cfg.reference_process
    noise_schedule = alg_cfg.noise_schedule
    loss_type = alg_cfg.loss_type

    target_xs = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))

    # Define initial and target density
    lambda_fn = lambda _: 1.0
    if reference_process == "pinned_brownian":  # Following PIS
        initial_dist = None  # actually, the initial distribution is Dirac delta at the origin
        aux_tuple = (dim, noise_schedule)
    elif reference_process in ["ou", "ou_dds"]:  # DIS or DDS
        initial_dist = distrax.MultivariateNormalDiag(
            jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
        )
        aux_tuple = (alg_cfg.init_std, initial_dist.log_prob)
        if reference_process == "ou_dds":
            alphas = cos_sq_fn_step_scheme(num_steps, noise_scale=alg_cfg.noise_scale)
            lambdas = 1 - (1 - alphas)[::-1].cumprod()[::-1]
            alpha_fn = lambda step: alphas[step]
            lambda_fn = lambda step: lambdas[step]
            aux_tuple = (*aux_tuple, alpha_fn, lambda_fn)
        else:
            aux_tuple = (*aux_tuple, noise_schedule)
    else:
        raise ValueError(f"Reference process {reference_process} not supported.")

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

    # Define the function to simulate subtrajectories with SMC & MCMC
    simulate_subtraj_partial = partial(
        batch_simulate_fwd_subtrajectories,
        batch_size=batch_size,
        initial_dist=initial_dist,
        target=target,
        num_steps=num_steps,
        num_subtrajs=n_chunks,
        sampling_configs=(
            reference_process,
            aux_tuple,
            alg_cfg.model.use_lp,
            alg_cfg.partial_energy,
            alg_cfg.beta_schedule,
        ),
        smc_configs=(
            smc_cfg.use,
            smc_cfg.resample_threshold,
            get_sampling_func(smc_cfg.sampling_method),
            smc_cfg.target_ess,
        ),
        mcmc_configs=(
            mcmc_cfg.use,
            mcmc_cfg.chain_length,
            mcmc_cfg.step_size,
            mcmc_cfg.n_burnin,
            mcmc_cfg.adapt,
            mcmc_cfg.target_acceptance_rate,
        ),
    )
    simulate_subtraj_jit = jax.jit(simulate_subtraj_partial)

    # Define the function to simulate the whole trajectory
    simulate_traj_partial = partial(
        rnd,
        reference_process=reference_process,
        aux_tuple=aux_tuple,
        target=target,
        num_steps=num_steps,
        use_lp=alg_cfg.model.use_lp,
        partial_energy=alg_cfg.partial_energy,
        beta_schedule=alg_cfg.beta_schedule,
        initial_dist=initial_dist,
    )

    if loss_type == "subtb":
        loss_fn_base = partial(loss_fn_subtb, n_chunks=n_chunks, logr_clip=alg_cfg.logr_clip)
    elif loss_type == "subtb_lambda":
        lambda_coef_mat = get_subtb_lambda_coefs(alg_cfg.lamda, num_steps)
        loss_fn_base = partial(
            loss_fn_subtb_lambda, lambda_coef_mat=lambda_coef_mat, logr_clip=alg_cfg.logr_clip
        )
    elif loss_type in ["tb_subtb", "lv_subtb"]:
        loss_fn_base = partial(
            loss_fn_joint,
            loss_type=loss_type,  # tb or lv
            n_chunks=n_chunks,
            subtb_weight=alg_cfg.subtb_weight,
            logr_clip=alg_cfg.logr_clip,
        )
    elif loss_type == "tb_subtb_lambda":
        lambda_coef_mat = get_subtb_lambda_coefs(alg_cfg.lamda, num_steps)
        loss_fn_base = partial(
            loss_fn_joint_lambda,
            loss_type=loss_type,  # tb or lv
            lambda_coef_mat=lambda_coef_mat,
        )
    else:
        raise ValueError(f"Loss type {loss_type} not supported.")

    # Define the function to be JIT-ed for FWD pass
    @jax.jit
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_fwd_grad_fn(key, model_state, params, invtemp=1.0):
        rnd_p = partial(simulate_traj_partial, batch_size=batch_size, prior_to_target=True)
        return loss_fn_base(key, model_state, params, rnd_partial=rnd_p, invtemp=invtemp)

    # Define the function to be JIT-ed for BWD pass
    @jax.jit
    @partial(jax.grad, argnums=2, has_aux=True)
    def loss_bwd_grad_fn(key, model_state, params, terminal_xs, log_rewards, invtemp=1.0):
        # prior_to_target=False, terminal_xs is now an argument
        rnd_p = partial(
            simulate_traj_partial,
            batch_size=batch_size,
            prior_to_target=False,
            terminal_xs=terminal_xs,
            log_rewards=log_rewards,
        )
        return loss_fn_base(key, model_state, params, rnd_partial=rnd_p, invtemp=invtemp)

    ### Prepare eval function
    eval_fn, logger = get_eval_fn(
        partial(simulate_traj_partial, batch_size=cfg.eval_samples), target, target_xs, cfg
    )
    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)

    ### Plot the True intermediate distributions
    if (
        cfg.use_wandb
        and getattr(target, "log_prob_t", None) is not None
        and reference_process == "ou_dds"  # TODO: support other reference processes
        and cfg.target.name in ["gaussian_mixture", "gaussian_mixture40"]  # TODO: support others
    ):
        true_vis_dict = visualise_true_intermediate_distribution(
            target.visualise,
            [i * (num_steps // n_chunks) for i in range(n_chunks + 1)],
            num_steps,
            reference_process,
            noise_schedule,
            lambda_fn,
            alg_cfg.init_std,
            target.log_prob,
            target.log_prob_t,
        )
        wandb.log(
            {
                key.replace("figures/", "figures_true/"): value
                for key, value in true_vis_dict.items()
            },
            step=0,
        )

    ### Prefill phase; initialise logZ with SMC-estimation
    if use_buffer and buffer_cfg.prefill_steps > 0:
        # logZ_estimates = []
        for _ in range(buffer_cfg.prefill_steps):
            key, key_gen = jax.random.split(key_gen)
            final_states, final_log_iws, _, _, _, _, end_state_log_fs, logZ_est = (
                simulate_subtraj_jit(key, model_state, model_state.params)
            )

            buffer_state = buffer.add(
                buffer_state,
                final_states,
                final_log_iws,
                end_state_log_fs[-1, :],
                jnp.zeros_like(final_log_iws),  # loss-prioritized buffer should not be used here
            )
    #         logZ_estimates.append(logZ_est)
    #     logZ_init = jax.nn.logsumexp(jnp.stack(logZ_estimates)) - jnp.log(buffer_cfg.prefill_steps)
    # else:
    #     key, key_gen = jax.random.split(key_gen)
    #     final_states, final_log_iws, _, _, _, _, end_state_log_fs, logZ_est = simulate_subtraj_jit(
    #         key, model_state, model_state.params
    #     )
    #     logZ_init = logZ_est
    # model_state.params["params"]["logZ"] = jnp.atleast_1d(logZ_init)
    # print(f"logZ_init: {logZ_init:.4f}")

    tb_losses = jnp.zeros(batch_size)  # to avoid error in wandb logging
    ### Training phase
    for it in range(alg_cfg.iters):
        invtemp = get_invtemp(
            it, alg_cfg.iters // 2, alg_cfg.init_invtemp, (alg_cfg.init_invtemp < 1.0)
        )

        # On-policy training with forward samples
        if not use_buffer or it % (buffer_cfg.bwd_to_fwd_ratio + 1) == 0:
            # Sample from model
            key, key_gen = jax.random.split(key_gen)
            grads, (trajectories, log_pbs_over_pfs, log_rewards, tb_losses, subtb_losses) = (
                loss_fwd_grad_fn(key, model_state, model_state.params, invtemp=invtemp)
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Add samples to buffer
            if use_buffer:
                buffer_state = buffer.add(
                    buffer_state,
                    trajectories[:, -1],
                    (log_pbs_over_pfs.sum(-1) + log_rewards),
                    log_rewards,
                    jnp.zeros_like(log_rewards),  # loss-prioritized buffer should not be used here
                )

        # Off-policy training with buffer samples
        else:
            # Sample terminal states using smc and store in buffer
            if alg_cfg.smc.use:
                key, key_gen = jax.random.split(key_gen)
                samples, final_log_iws, _, _, _, _, end_state_log_fs, _ = simulate_subtraj_jit(
                    key, model_state, model_state.params
                )
                log_rewards = end_state_log_fs[-1, :]

                buffer_state = buffer.add(
                    buffer_state,
                    samples,
                    final_log_iws,
                    log_rewards,
                    jnp.zeros_like(final_log_iws),
                )

            # Sample terminal states from buffer
            key, key_gen = jax.random.split(key_gen)
            samples, log_rewards, indices = buffer.sample(buffer_state, key, batch_size)

            # Get grads with the off-policy samples
            key, key_gen = jax.random.split(key_gen)
            grads, (_, log_pbs_over_pfs, log_rewards, tb_losses, subtb_losses) = loss_bwd_grad_fn(
                key, model_state, model_state.params, samples, log_rewards, invtemp=invtemp
            )
            model_state = model_state.apply_gradients(grads=grads)

            # Update scores in buffer if needed
            if buffer_cfg.update_score:
                buffer_state = buffer.update_priority(
                    buffer_state,
                    indices,
                    (log_pbs_over_pfs.sum(-1) + log_rewards),
                    log_rewards,
                    jnp.zeros_like(log_rewards),  # loss-prioritized buffer should not be used here
                )

        if cfg.use_wandb:
            log_dict = {
                "tb_loss": jnp.mean(tb_losses),
                "subtb_loss": jnp.mean(subtb_losses.mean(-1)),
            }
            if "logZ" in model_state.params["params"]:
                log_dict["logZ_learned"] = model_state.params["params"]["logZ"]
            wandb.log(log_dict, step=it)

        if (it % eval_freq == 0) or (it == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(it)
            logger["stats/nfe"].append((it + 1) * batch_size)  # FIXME

            logger.update(eval_fn(model_state, key))

            # Visualize intermediate distributions (learned flows)
            if cfg.use_wandb and cfg.target.name in ["gaussian_mixture", "gaussian_mixture40"]:
                beta_fn = get_beta_fn(
                    model_state.params, alg_cfg.beta_schedule, num_steps, lambda_fn
                )
                vis_dict = visualise_intermediate_distribution(
                    target.visualise,
                    [i * (num_steps // n_chunks) for i in range(n_chunks)],
                    num_steps,
                    None,  # trajectories,
                    model_state,
                    alg_cfg.partial_energy,
                    batch_size,
                    reference_process,
                    noise_schedule,
                    beta_fn,
                    initial_dist,
                    target.log_prob,
                )
                logger.update(vis_dict)

            # Evaluate buffer samples
            if use_buffer:
                from eval import discrepancies

                buffer_xs, _, _ = buffer.sample(buffer_state, key, cfg.eval_samples)

                for d in cfg.discrepancies:
                    if logger.get(f"discrepancies/buffer_samples_{d}", None) is None:
                        logger[f"discrepancies/buffer_samples_{d}"] = []
                    logger[f"discrepancies/buffer_samples_{d}"].append(
                        getattr(discrepancies, f"compute_{d}")(target_xs, buffer_xs, cfg)
                        if target_xs is not None
                        else jnp.inf
                    )
                if cfg.use_wandb:
                    vis_dict = target.visualise(samples=buffer_xs)
                    logger.update(
                        {f"{key}_buffer_samples": value for key, value in vis_dict.items()}
                    )

            # # Evaluate smc samples
            # if alg_cfg.smc.use:
            #     from eval import discrepancies

            #     smc_xs, _, _, _, _, _, _, _ = simulate_subtraj_fwd(
            #         key, model_state, model_state.params
            #     )

            #     for d in cfg.discrepancies:
            #         if logger.get(f"discrepancies/smc_samples_{d}", None) is None:
            #             logger[f"discrepancies/smc_samples_{d}"] = []
            #         logger[f"discrepancies/smc_samples_{d}"].append(
            #             getattr(discrepancies, f"compute_{d}")(target_xs, smc_xs, cfg)
            #             if target_xs is not None
            #             else jnp.inf
            #         )
            #     if cfg.use_wandb:
            #         vis_dict = target.visualise(samples=smc_xs)
            #         logger.update({f"{key}_smc_samples": value for key, value in vis_dict.items()})

            print_results(it, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger), step=it)
