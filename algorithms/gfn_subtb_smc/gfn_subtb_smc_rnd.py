from functools import partial
from typing import Callable, Literal

import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax.scipy.special import logsumexp

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.gfn_tb.gfn_tb_rnd import sample_kernel, log_prob_kernel
from algorithms.gfn_tb.sampling_utils import binary_search_smoothing, ess
from algorithms.gfn_subtb.gfn_subtb_rnd import (
    get_beta_fn,
    get_flow_bias,
    ref_log_prob_pinned_brownian,
)
from algorithms.gfn_subtb_smc.mcmcs import mala
from targets.base_target import Target


### per-step RND functions ###


def get_log_f(
    state: Array,
    model_state: TrainState,
    params: ModelParams,
    step: Array,
    num_steps: int,
    partial_energy: bool,
    ref_log_prob_fn: Callable[[Array], Array],
    target_log_prob_fn: Callable[[Array], Array],
    beta_fn: Callable[[int], float],
):
    def get_log_f_intermediate(state):
        _, log_f = model_state.apply_fn(
            params, state, (step / num_steps) * jnp.ones(1), jnp.zeros(state.shape[0])
        )  # langevin doesn't affect the _log_f
        if partial_energy:
            log_f = log_f + get_flow_bias(
                beta_fn(step), ref_log_prob_fn(state), target_log_prob_fn(state)
            )
        return log_f

    return jax.lax.cond(
        jnp.equal(step, num_steps), target_log_prob_fn, get_log_f_intermediate, operand=state
    )


def per_sample_subtraj_rnd_pinned_brownian(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    input_state: Array,
    aux_tuple: tuple,
    target: Target,
    num_steps: int,
    timestep_tup: tuple[int, int],  # (start_step, subtraj_length)
    use_lp: bool,
    partial_energy: bool,
    beta_schedule: Literal["learnt", "linear", "cosine"],
):
    dim, noise_schedule = aux_tuple
    assert beta_schedule != "cosine", "Cosine beta schedule not supported for pinned_brownian."
    beta_fn = get_beta_fn(params, beta_schedule, num_steps)
    dt = 1.0 / num_steps

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        sigma_t = noise_schedule(step)
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s = jax.lax.stop_gradient(s)

        log_prob = jnp.array(0.0)
        langevin = jnp.zeros(dim)
        if use_lp:
            log_prob, langevin = jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
        elif partial_energy:
            log_prob = target.log_prob(s)

        log_f_bias = jnp.array(0.0)
        if partial_energy:
            ref_log_prob = jax.lax.cond(
                step == 0,
                lambda _: 0.0,
                lambda args: ref_log_prob_pinned_brownian(*args),
                operand=(s, t, sigma_t),
            )
            log_f_bias = get_flow_bias(beta_fn(step), ref_log_prob, log_prob)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        log_f = log_f + log_f_bias

        # Euler-Maruyama integration of the SDE
        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute backward SDE components
        shrink = (t_next - dt) / t_next  # == t / t_next when uniform time steps
        bwd_mean = shrink * s_next
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt)
        bwd_log_prob = jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(*args),
            operand=(s, bwd_mean, bwd_scale),
        )

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    start_step, subtraj_length = timestep_tup
    end_step = start_step + subtraj_length

    start_state = input_state
    aux = (start_state, key)
    aux, per_step_output = jax.lax.scan(
        simulate_prior_to_target, aux, start_step + jnp.arange(subtraj_length)
    )
    end_state, _ = aux
    subtrajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output

    def end_state_ref_log_prob(s):
        sigma_end = noise_schedule(end_step)
        _end_step = end_step.astype(jnp.float32)
        t_end = _end_step / num_steps
        return ref_log_prob_pinned_brownian(s, t_end, sigma_end)

    end_state_log_f = get_log_f(
        state=end_state,
        model_state=model_state,
        params=params,
        step=end_step,
        num_steps=num_steps,
        partial_energy=partial_energy,
        ref_log_prob_fn=end_state_ref_log_prob,
        target_log_prob_fn=target.log_prob,
        beta_fn=beta_fn,
    )
    return end_state, subtrajectory, fwd_log_prob, bwd_log_prob, log_f, end_state_log_f


def per_sample_subtraj_rnd_ou(*args, **kwargs):
    raise NotImplementedError("OU reference process not implemented yet.")


def per_sample_subtraj_rnd_ou_dds(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    input_state: Array,
    aux_tuple: tuple,
    target: Target,
    num_steps: int,
    timestep_tup: tuple[int, int],  # (start_step, subtraj_length)
    use_lp: bool,
    partial_energy: bool,
    beta_schedule: Literal["learnt", "linear", "cosine"],
):
    init_std, init_log_prob_fn, alpha_fn, lambda_fn = aux_tuple
    beta_fn = get_beta_fn(params, beta_schedule, num_steps, lambda_fn)

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        t = step / num_steps

        s = jax.lax.stop_gradient(s)

        # Compute forward SDE components
        log_prob = jnp.array(0.0)
        langevin = jnp.zeros(s.shape[0])
        if use_lp:
            log_prob, langevin = jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
        elif partial_energy:
            log_prob = target.log_prob(s)

        log_f_bias = jnp.array(0.0)
        if partial_energy:
            log_f_bias = get_flow_bias(beta_fn(step), init_log_prob_fn(s), log_prob)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        log_f = log_f + log_f_bias

        # Exponential integration of the SDE
        sqrt_at = jnp.clip(jnp.sqrt(alpha_fn(step)), 0, 1)
        sqrt_1_minus_at = jnp.sqrt(1 - sqrt_at**2)
        fwd_mean = sqrt_1_minus_at * s + sqrt_at**2 * model_output
        fwd_scale = sqrt_at * init_std
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute backward SDE components
        bwd_mean = sqrt_1_minus_at * s_next
        bwd_scale = sqrt_at * init_std
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale)

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    start_step, subtraj_length = timestep_tup
    end_step = start_step + subtraj_length

    start_state = input_state
    aux = (start_state, key)
    aux, per_step_output = jax.lax.scan(
        simulate_prior_to_target, aux, start_step + jnp.arange(subtraj_length)
    )
    end_state, _ = aux
    subtrajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output

    end_state_log_f = get_log_f(
        state=end_state,
        model_state=model_state,
        params=params,
        step=end_step,
        num_steps=num_steps,
        partial_energy=partial_energy,
        ref_log_prob_fn=init_log_prob_fn,
        target_log_prob_fn=target.log_prob,
        beta_fn=beta_fn,
    )
    return end_state, subtrajectory, fwd_log_prob, bwd_log_prob, log_f, end_state_log_f


def resampling(
    key: RandomKey,
    states: Array,
    log_iws: Array,
    sampling_func: Callable[[RandomKey, Array, int, bool], Array],
    target_ess: float,
):
    tempered_log_iws, temp = binary_search_smoothing(log_iws, target_ess)
    indices = sampling_func(
        key, jax.nn.softmax(tempered_log_iws, axis=0), log_iws.shape[0], replacement=True
    )
    resampled_states = states[indices]
    resampled_log_iws = log_iws[indices] * (1 - 1 / temp)
    return resampled_states, resampled_log_iws


### simulate ###
def batch_simulate_fwd_subtrajectories(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    batch_size: int,
    initial_dist: distrax.Distribution | None,
    target: Target,
    num_steps: int,
    num_subtrajs: int,
    sampling_configs: tuple[
        Literal["pinned_brownian", "ou", "ou_dds"],  # reference_process
        tuple,  # aux_tuple; for ou_dds, (init_std, initial_dist.log_prob, alpha_fn, lambda_fn)
        bool,  # use_lp
        bool,  # partial_energy
        Literal["learnt", "linear", "cosine"],  # beta_schedule
    ],
    smc_configs: tuple[
        bool,  # use
        float,  # resample_threshold
        Callable[[RandomKey, Array, int, bool], Array],  # sampling_func
        float,  # target_ess in [0, 1]
    ],
    mcmc_configs: tuple[
        bool,  # use
        int,  # chain_length
        float,  # step_size
        int,  # n_burnin
        bool,  # adapt
        float,  # target_acceptance_rate
    ],
):
    assert num_steps % num_subtrajs == 0
    subtraj_length = num_steps // num_subtrajs

    reference_process, aux_tuple, use_lp, partial_energy, beta_schedule = sampling_configs
    use_resampling, resample_threshold, sampling_func, target_ess = smc_configs
    use_mcmc, chain_length, step_size, n_burnin, adapt, target_acceptance_rate = mcmc_configs
    init_log_prob_fn = lambda s: (
        initial_dist.log_prob(s) if initial_dist is not None else jnp.zeros(s.shape[0])
    )

    log_f_fn_partial = None
    if use_mcmc:
        lambda_fn = aux_tuple[3] if reference_process == "ou_dds" else None
        beta_fn = get_beta_fn(model_state.params, beta_schedule, num_steps, lambda_fn)

        log_f_fn_partial = partial(
            get_log_f,
            model_state=model_state,
            params=model_state.params,
            num_steps=num_steps,
            partial_energy=partial_energy,
            init_log_prob_fn=init_log_prob_fn,
            target_log_prob_fn=target.log_prob,
            beta_fn=beta_fn,
        )

    ### subtrajectory step functions ###
    def batch_simulate_subtrajectory(step_input, per_step_input):
        states, log_iws, init_fwd_log_probs, key_gen = step_input
        start_step = per_step_input
        end_step = start_step + subtraj_length

        ## vectorized subtrajectory sampling
        key, key_gen = jax.random.split(key_gen)
        keys = jax.random.split(key, num=batch_size)
        per_sample_fn = {
            "pinned_brownian": per_sample_subtraj_rnd_pinned_brownian,
            "ou": per_sample_subtraj_rnd_ou,
            "ou_dds": per_sample_subtraj_rnd_ou_dds,
        }[reference_process]

        (
            next_states,
            subtrajectories,
            fwd_log_probs,
            bwd_log_probs,
            log_fs,
            end_state_log_fs,
        ) = jax.vmap(
            per_sample_fn,
            in_axes=(0, None, None, 0, None, None, None, None, None, None, None),
        )(
            keys,
            model_state,
            params,
            states,
            aux_tuple,
            target,
            num_steps,
            (start_step, subtraj_length),
            use_lp,
            partial_energy,
            beta_schedule,
        )

        start_state_log_fs = jax.lax.cond(
            start_step == 0,
            lambda _: init_fwd_log_probs,
            lambda _: log_fs[:, 0],
            operand=None,
        )
        next_log_iws = log_iws + jax.lax.stop_gradient(
            end_state_log_fs + bwd_log_probs.sum(-1) - start_state_log_fs - fwd_log_probs.sum(-1)
        )

        ## Do resampling with the adaptive tempering
        key, key_gen = jax.random.split(key_gen)
        logZ_ratio = logsumexp(next_log_iws)  # log(\hat{Z_t / Z_{t_prev}})
        if use_resampling and target_ess < 1.0:  # if target_ess is 1.0, no need to resample
            normalized_ess = ess(log_iws=next_log_iws) / batch_size
            next_states, next_log_iws = jax.lax.cond(
                jnp.logical_and(
                    normalized_ess < resample_threshold,
                    start_step + subtraj_length != num_steps,  # don't resample at the last step
                ),
                # normalized_ess < resample_threshold,
                lambda args: resampling(args[0], args[1], args[2], sampling_func, target_ess),
                lambda args: (args[1], args[2]),
                (key, next_states, next_log_iws),
            )
        next_log_iws = jax.nn.log_softmax(next_log_iws, axis=0)

        ## Do MCMC (Rejuvenation)
        key, key_gen = jax.random.split(key_gen)
        if use_mcmc:
            next_states, new_log_fs, _ = mala(
                key,
                next_states,
                partial(log_f_fn_partial, step=end_step),
                chain_length,
                step_size,
                n_burnin,
                adapt,
                target_acceptance_rate,
            )
            end_state_log_fs = jax.lax.cond(
                end_step == num_steps,
                lambda _: new_log_fs,
                lambda _: end_state_log_fs,
                operand=None,
            )

        ## Return outputs
        next_step_input = (next_states, next_log_iws, init_fwd_log_probs, key_gen)
        per_subtraj_outputs = (
            subtrajectories,
            fwd_log_probs,
            bwd_log_probs,
            log_fs,
            end_state_log_fs,
            logZ_ratio,
        )
        return next_step_input, per_subtraj_outputs

    # Define initial state and per step inputs
    key, key_gen = jax.random.split(key)
    if initial_dist is not None:
        init_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        init_states = jnp.zeros((batch_size, target.dim))
    init_log_iws = -jnp.log(batch_size) * jnp.ones(batch_size)
    init_fwd_log_probs = init_log_prob_fn(init_states)
    init_input = (init_states, init_log_iws, init_fwd_log_probs, key_gen)
    per_subtraj_inputs = jnp.arange(0, num_steps, subtraj_length)

    # Scan over subtrajectories
    final_outputs, per_subtraj_outputs = jax.lax.scan(
        batch_simulate_subtrajectory, init_input, per_subtraj_inputs
    )
    final_states, final_log_iws, _, _ = final_outputs
    # final_states.shape == (batch_size, dim)
    # final_iws.shape == (batch_size,)
    subtrajectories, fwd_log_probs, bwd_log_probs, log_fs, end_state_log_fs, logZ_ratio = (
        per_subtraj_outputs
    )
    fwd_log_probs = fwd_log_probs.at[0, :, 0].set(fwd_log_probs[0, :, 0] + init_fwd_log_probs)
    # subtrajectories.shape == (#subtrajs, batch_size, subtraj_length, dim)
    # fwd_log_probs, bwd_log_probs, log_fs have shape (#subtrajs, batch_size, subtraj_length)
    # end_state_log_fs have shape (#subtrajs, batch_size)
    # logZ_ratio have shape (#subtrajs,)

    logZ_est = logZ_ratio.sum()
    final_log_iws = final_log_iws + logZ_est + jnp.log(batch_size)
    # jax.debug.print("logZ_est w/ SMC: {logZ_est}", logZ_est=logZ_est)

    return (
        final_states,
        final_log_iws,
        subtrajectories,
        fwd_log_probs,
        bwd_log_probs,
        log_fs,
        end_state_log_fs,
        logZ_est,
    )


def loss_fn_subtraj(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    simulate_subtraj: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
):
    (
        final_states,
        final_iws,
        subtrajectories,
        fwd_log_probs,
        bwd_log_probs,
        log_fs,
        end_state_log_fs,
    ) = simulate_subtraj(key, model_state, params)

    log_rewards = end_state_log_fs[-1, :]
    log_rewards = jnp.where(
        log_rewards > logr_clip,
        log_rewards,
        logr_clip - jnp.log(logr_clip + log_rewards),
    )

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    end_state_log_fs = end_state_log_fs.at[-1, :].set(log_rewards * invtemp)

    log_fs = log_fs.at[0, :, 0].set(params["params"]["logZ"])
    log_fs = jnp.concatenate([log_fs, end_state_log_fs[:, :, None]], axis=2)

    # db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    # subtb_discrepancy1 = db_discrepancy.reshape(bs, n_subtrajs, -1).sum(-1)
    # The below is equivalent to the above lines but avoids numerical instability.
    subtb_discrepancy = log_fs[:, :, 0] + log_pfs_over_pbs.sum(-1) - log_fs[:, :, -1]

    if huber_delta is not None:
        subtb_losses = jnp.where(
            jnp.abs(subtb_discrepancy) <= huber_delta,
            jnp.square(subtb_discrepancy),
            huber_delta * jnp.abs(subtb_discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        subtb_losses = jnp.square(subtb_discrepancy)

    return jnp.mean(subtb_losses.mean(0)), (
        final_states,
        jax.lax.stop_gradient(final_iws),
        subtrajectories,
        log_rewards,
        jax.lax.stop_gradient(subtb_losses),
    )
