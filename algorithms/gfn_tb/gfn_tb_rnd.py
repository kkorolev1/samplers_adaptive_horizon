from typing import Callable, Literal

import distrax
import jax
import jax.numpy as jnp
import numpyro.distributions as npdist
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.gfn_tb.sampling_utils import binary_search_smoothing


def sample_kernel(key_gen, mean, scale):
    key, key_gen = jax.random.split(key_gen)
    eps = jnp.clip(jax.random.normal(key, shape=(mean.shape[0],)), -4.0, 4.0)
    return mean + scale * eps, key_gen


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def per_sample_rnd_pinned_brownian(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    prior_to_target=True,
):
    dim, noise_schedule = aux_tuple
    dt = 1.0 / num_steps

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        sigma_t = noise_schedule(step)
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s = jax.lax.stop_gradient(s)

        # Compute SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))

        else:
            langevin = jnp.zeros(dim)
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)

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
        per_step_output = (s, fwd_log_prob, bwd_log_prob)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        sigma_t = noise_schedule(step)
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s_next = jax.lax.stop_gradient(s_next)

        # Compute backward SDE components
        shrink = (t_next - dt) / t_next
        bwd_mean = shrink * s_next
        bwd_scale = sigma_t * jnp.sqrt(shrink * dt)
        s, key_gen = jax.lax.cond(
            step == 0,
            lambda _: (jnp.zeros_like(s_next), key_gen),
            lambda args: sample_kernel(*args),
            operand=(key_gen, bwd_mean, bwd_scale),
        )
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(*args),
            operand=(s, bwd_mean, bwd_scale),
        )

        # Compute forward SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
        else:
            langevin = jnp.zeros(dim)
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)

        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute importance weight increment
        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob)
        return next_state, per_step_output

    if prior_to_target:
        init_x = input_state
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(num_steps)
        )
        terminal_x, _ = aux
    else:
        terminal_x = input_state
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )

    trajectory, fwd_log_prob, bwd_log_prob = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob


def per_sample_rnd_ou(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    prior_to_target=True,
):
    raise NotImplementedError("OU reference process not implemented yet.")


def per_sample_rnd_ou_dds(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    prior_to_target=True,
):
    init_std, alpha_fn = aux_tuple

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        t = step / num_steps

        s = jax.lax.stop_gradient(s)

        # Compute forward SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
        else:
            langevin = jnp.zeros(s.shape[0])
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)

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
        per_step_output = (s, fwd_log_prob, bwd_log_prob)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        t = step / num_steps

        s_next = jax.lax.stop_gradient(s_next)

        # Compute backward SDE components
        sqrt_at_next = jnp.clip(jnp.sqrt(alpha_fn(step)), 0, 1)
        sqrt_1_minus_at_next = jnp.sqrt(1 - sqrt_at_next**2)
        bwd_mean = sqrt_1_minus_at_next * s_next
        bwd_scale = sqrt_at_next * init_std
        s, key_gen = sample_kernel(key_gen, bwd_mean, bwd_scale)
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale)

        # Compute forward SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
        else:
            langevin = jnp.zeros(s.shape[0])
        model_output, _ = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)

        fwd_mean = sqrt_1_minus_at_next * s + sqrt_at_next**2 * model_output
        fwd_scale = sqrt_at_next * init_std
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Return next state and per-step output
        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob)
        return next_state, per_step_output

    if prior_to_target:
        init_x = input_state
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(num_steps)
        )
        terminal_x, _ = aux
    else:
        terminal_x = input_state
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )

    trajectory, fwd_log_prob, bwd_log_prob = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob


def rnd(
    key_gen,
    model_state,
    params,
    reference_process: Literal["pinned_brownian", "ou", "ou_dds"],
    batch_size,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    prior_to_target=True,
    initial_dist: distrax.Distribution | None = None,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
):
    if prior_to_target:
        if initial_dist is None:  # pinned_brownian
            input_states = jnp.zeros((batch_size, target.dim))
        else:
            key, key_gen = jax.random.split(key_gen)
            input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    per_sample_fn = {
        "pinned_brownian": per_sample_rnd_pinned_brownian,
        "ou": per_sample_rnd_ou,
        "ou_dds": per_sample_rnd_ou_dds,
    }[reference_process]

    terminal_xs, trajectories, fwd_log_probs, bwd_log_probs = jax.vmap(
        per_sample_fn,
        in_axes=(0, None, None, 0, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps,
        use_lp,
        prior_to_target,
    )

    if not prior_to_target:
        trajectories = trajectories[:, ::-1]
        fwd_log_probs = fwd_log_probs[:, ::-1]
        bwd_log_probs = bwd_log_probs[:, ::-1]

    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)

    if initial_dist is None:  # pinned_brownian
        init_fwd_log_probs = jnp.zeros(batch_size)
    else:
        init_fwd_log_probs = initial_dist.log_prob(trajectories[:, 0])

    if log_rewards is None:
        log_rewards = target.log_prob(terminal_xs)

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    return (
        trajectories[:, -1],
        log_pfs_over_pbs.sum(1) + init_fwd_log_probs,
        jnp.zeros_like(log_rewards),
        -log_rewards,
    )


def loss_fn(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[
        [RandomKey, TrainState, ModelParams], tuple[Array, Array, Array, Array]
    ],
    loss_type: Literal["tb", "lv"],
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
    importance_weighting: bool = False,
    target_ess: float = 0.0,
):
    terminal_xs, running_costs, _, terminal_costs = rnd_partial(
        key, model_state, params
    )
    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    log_ratio = running_costs - log_rewards * invtemp  # log_pfs - log_pbs - log_rewards
    if loss_type == "tb":
        losses = log_ratio + params["params"]["logZ"]
    else:  # loss_type == "lv"
        losses = log_ratio - jnp.mean(log_ratio)

    if huber_delta is not None:
        losses = jnp.where(
            jnp.abs(losses) <= huber_delta,
            jnp.square(losses),
            huber_delta * jnp.abs(losses) - 0.5 * huber_delta**2,
        )
    else:
        losses = jnp.square(losses)

    if importance_weighting:
        smoothed_log_iws, _ = binary_search_smoothing(-log_ratio, target_ess)
        normalized_iws = jax.nn.softmax(smoothed_log_iws, axis=-1)
        loss = jnp.sum(losses * normalized_iws)
    else:
        loss = jnp.mean(losses)

    return loss, (
        terminal_xs,
        jax.lax.stop_gradient(-running_costs),  # log_pbs - log_pfs
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(losses),  # losses
    )
