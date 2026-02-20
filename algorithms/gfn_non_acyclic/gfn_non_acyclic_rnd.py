from typing import Callable, Literal

import distrax
import jax
import jax.nn as nn
import jax.numpy as jnp
from functools import partial
import numpyro.distributions as npdist
from flax.training.train_state import TrainState
import equinox

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.gfn_non_acyclic.sampling_utils import binary_search_smoothing


def sample_kernel(key_gen, mean, scale):
    key, key_gen = jax.random.split(key_gen)
    eps = jnp.clip(jax.random.normal(key, shape=(mean.shape[0],)), -4.0, 4.0)
    return mean + scale * eps, key_gen


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def clip_log_reward(log_reward, clip_value=-1e5):
    return jnp.where(
        log_reward > clip_value,
        log_reward,
        clip_value - jnp.log(clip_value - log_reward),
    )


def per_sample_rnd_no_term(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    prior_to_target=True,
):
    (logr_clip,) = aux_tuple

    # @jax.checkpoint
    def model_forward(s, log_reward, langevin):
        return model_state.apply_fn(params, s, log_reward, langevin, predict_fwd=True)

    # @jax.checkpoint
    def model_backward(s_next):
        return model_state.apply_fn(params, s_next, predict_bwd=True)

    # @jax.checkpoint
    def compute_log_reward_and_langevin(s):
        return jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        s = jax.lax.stop_gradient(s)

        log_reward, langevin = compute_log_reward_and_langevin(s)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s, log_reward, langevin
        )
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + nn.log_sigmoid(
            -fwd_clf_logits
        )

        bwd_clf_logits, bwd_mean, bwd_scale = model_backward(s_next)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale) + jax.nn.log_sigmoid(
            -bwd_clf_logits
        )

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        s_next = jax.lax.stop_gradient(s_next)
        bwd_clf_logits, bwd_mean, bwd_scale = model_backward(s_next)
        s, key_gen = sample_kernel(key_gen, bwd_mean, bwd_scale)
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale) + jax.nn.log_sigmoid(
            -bwd_clf_logits
        )

        log_reward, langevin = compute_log_reward_and_langevin(s)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s, log_reward, langevin
        )
        fwd_log_prob = log_prob_kernel(
            s_next, fwd_mean, fwd_scale
        ) + jax.nn.log_sigmoid(-fwd_clf_logits)

        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
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
            simulate_target_to_prior, aux, jnp.arange(num_steps)
        )
    trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob, log_f


def rnd_no_term(
    key_gen,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    prior_to_target=True,
    initial_dist: distrax.Distribution | None = None,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
):
    if prior_to_target:
        key, key_gen = jax.random.split(key_gen)
        input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    terminal_xs, trajectories, fwd_log_probs, bwd_log_probs, log_fs = jax.vmap(
        per_sample_rnd_no_term,
        in_axes=(0, None, None, 0, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps - 1,
        prior_to_target,
    )
    if not prior_to_target:
        trajectories = trajectories[:, ::-1]
        fwd_log_probs = fwd_log_probs[:, ::-1]
        bwd_log_probs = bwd_log_probs[:, ::-1]
        log_fs = log_fs[:, ::-1]

    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)

    if initial_dist is None:  # pinned_brownian
        init_fwd_log_probs = jnp.zeros(batch_size)
    else:
        init_fwd_log_probs = initial_dist.log_prob(trajectories[:, 0])

    if log_rewards is None:
        log_rewards = target.log_prob(terminal_xs)

    # We need to calculate log flow for the last continuous xs
    terminal_xs = jax.lax.stop_gradient(terminal_xs)
    *_, terminal_log_fs = model_state.apply_fn(
        params, terminal_xs, log_rewards, predict_fwd=True
    )

    # We need to calculate bwd_log_prob for the first continuous xs
    init_xs = jax.lax.stop_gradient(trajectories[:, 0])
    bwd_clf_logits, *_ = model_state.apply_fn(params, init_xs, predict_bwd=True)

    fwd_log_probs = jnp.concatenate(
        [init_fwd_log_probs[:, None], fwd_log_probs], axis=1
    )
    bwd_log_probs = jnp.concatenate(
        [nn.log_sigmoid(bwd_clf_logits)[:, None], bwd_log_probs], axis=1
    )
    logZ = params["params"]["logZ"]
    log_fs = jnp.concatenate(
        [
            jnp.zeros_like(terminal_log_fs)[:, None],  # set to logZ later
            log_fs,
            terminal_log_fs[:, None],
        ],
        axis=1,
    )
    log_fs = log_fs.at[:, 0].set(logZ)

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs

    return (
        trajectories,
        log_pfs_over_pbs.sum(1),  # running costs
        jnp.zeros_like(log_rewards),  # stochastic costs
        -log_rewards,  # terminal costs
        (num_steps + 1) * jnp.ones((batch_size,), dtype=int),
        log_pfs_over_pbs,
        log_fs,
    )


def per_sample_rnd_with_term(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    initial_dist,
    prior_to_target=True,
):
    (logr_clip,) = aux_tuple

    # @jax.checkpoint
    def model_forward(s, log_reward, langevin, force_stop=False):
        return model_state.apply_fn(
            params, s, log_reward, langevin, predict_fwd=True, force_stop=force_stop
        )

    # @jax.checkpoint
    def model_backward(s_next, force_stop=False):
        return model_state.apply_fn(
            params, s_next, predict_bwd=True, force_stop=force_stop
        )

    # @jax.checkpoint
    def compute_log_reward_and_langevin(s):
        return jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))

    def cond_fun(carry):
        state, _ = carry
        s, is_terminal, key_gen, step = state
        return (jnp.any(~is_terminal)) & (step < num_steps)

    def simulate_prior_to_target(carry, force_stop=False):
        state, state_hist = carry
        s, is_terminal, key_gen, step = state
        trajectories, terminals_mask, fwd_log_probs, bwd_log_probs, log_fs = state_hist

        s = jax.lax.stop_gradient(s)
        log_reward, langevin = compute_log_reward_and_langevin(s)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s,
            log_reward,
            langevin,
            force_stop=force_stop,
        )
        key, key_gen = jax.random.split(key_gen)
        is_terminal_next = is_terminal | jax.random.bernoulli(
            key, nn.sigmoid(fwd_clf_logits)
        )
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + nn.log_sigmoid(
            -fwd_clf_logits
        )
        fwd_log_prob = jnp.where(
            is_terminal_next, nn.log_sigmoid(fwd_clf_logits), fwd_log_prob
        )
        fwd_log_prob = jnp.where(
            is_terminal, jnp.zeros_like(fwd_clf_logits), fwd_log_prob
        )
        bwd_clf_logits, bwd_mean, bwd_scale = model_backward(s_next)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale) + jax.nn.log_sigmoid(
            -bwd_clf_logits
        )
        bwd_log_prob = jnp.where(is_terminal_next, log_reward, bwd_log_prob)
        bwd_log_prob = jnp.where(
            is_terminal, jnp.zeros_like(bwd_log_prob), bwd_log_prob
        )

        log_f = jnp.where(is_terminal, jnp.zeros_like(log_f), log_f)
        s_next = jnp.where(is_terminal_next, jnp.zeros_like(s_next), s_next)

        trajectories = trajectories.at[step].set(s)
        terminals_mask = terminals_mask.at[step].set(is_terminal)
        fwd_log_probs = fwd_log_probs.at[step].set(fwd_log_prob)
        bwd_log_probs = bwd_log_probs.at[step].set(bwd_log_prob)
        log_fs = log_fs.at[step].set(log_f)

        state_next = (s_next, is_terminal_next, key_gen, step + 1)
        state_hist = (
            trajectories,
            terminals_mask,
            fwd_log_probs,
            bwd_log_probs,
            log_fs,
        )
        return (state_next, state_hist)

    def simulate_target_to_prior(carry, force_stop=False):
        state, state_hist = carry
        s_next, is_terminal_next, key_gen, step = state
        trajectories, terminals_mask, fwd_log_probs, bwd_log_probs, log_fs = state_hist
        s_next = jax.lax.stop_gradient(s_next)
        bwd_clf_logits, bwd_mean, bwd_scale = model_backward(
            s_next,
            force_stop=force_stop,
        )
        key, key_gen = jax.random.split(key_gen)
        is_terminal = is_terminal_next | jax.random.bernoulli(
            key, nn.sigmoid(bwd_clf_logits)
        )
        s, key_gen = sample_kernel(key_gen, bwd_mean, bwd_scale)
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale) + nn.log_sigmoid(
            -bwd_clf_logits
        )
        bwd_log_prob = jnp.where(
            is_terminal, nn.log_sigmoid(bwd_clf_logits), bwd_log_prob
        )
        bwd_log_prob = jnp.where(
            is_terminal_next, jnp.zeros_like(bwd_clf_logits), bwd_log_prob
        )

        log_reward, langevin = compute_log_reward_and_langevin(s)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s, log_reward, langevin
        )

        fwd_log_prob = log_prob_kernel(
            s_next, fwd_mean, fwd_scale
        ) + jax.nn.log_sigmoid(-fwd_clf_logits)
        fwd_log_prob = jnp.where(
            is_terminal, initial_dist.log_prob(s_next), fwd_log_prob
        )
        fwd_log_prob = jnp.where(
            is_terminal_next, jnp.zeros_like(fwd_log_prob), fwd_log_prob
        )

        log_f = jnp.where(is_terminal, jnp.zeros_like(log_f), log_f)
        s = jnp.where(is_terminal, jnp.zeros_like(s), s)

        trajectories = trajectories.at[step].set(s)
        terminals_mask = terminals_mask.at[step].set(is_terminal)
        fwd_log_probs = fwd_log_probs.at[step].set(fwd_log_prob)
        bwd_log_probs = bwd_log_probs.at[step].set(bwd_log_prob)
        log_fs = log_fs.at[step].set(log_f)

        state_next = (s, is_terminal, key_gen, step + 1)
        state_hist = (
            trajectories,
            terminals_mask,
            fwd_log_probs,
            bwd_log_probs,
            log_fs,
        )
        return (state_next, state_hist)

    d = input_state.shape[-1]
    trajectories = jnp.zeros((num_steps + 1, d))
    terminals_mask = jnp.ones((num_steps + 1,), dtype=bool)
    fwd_log_probs = jnp.zeros((num_steps + 1,))
    bwd_log_probs = jnp.zeros((num_steps + 1,))
    log_fs = jnp.zeros((num_steps + 1,))
    state_hist = (trajectories, terminals_mask, fwd_log_probs, bwd_log_probs, log_fs)

    if prior_to_target:
        state_init = (input_state, jnp.array(False), key, 0)
        carry = (state_init, state_hist)
        carry = equinox.internal.while_loop(
            cond_fun,
            simulate_prior_to_target,
            carry,
            max_steps=num_steps,
            kind="checkpointed",
        )
        carry = simulate_prior_to_target(carry, force_stop=True)
    else:
        state_init = (input_state, jnp.array(False), key, 0)
        carry = (state_init, state_hist)
        carry = equinox.internal.while_loop(
            cond_fun,
            simulate_target_to_prior,
            carry,
            max_steps=num_steps,
            kind="checkpointed",
        )
        carry = simulate_target_to_prior(carry, force_stop=True)

    state, state_hist = carry
    return state_hist


def rnd_with_term(
    key_gen,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    prior_to_target=True,
    initial_dist: distrax.Distribution | None = None,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
):
    if prior_to_target:
        key, key_gen = jax.random.split(key_gen)
        input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    trajectories, terminals_mask, fwd_log_probs, bwd_log_probs, log_fs = jax.vmap(
        per_sample_rnd_with_term,
        in_axes=(0, None, None, 0, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps - 2 if prior_to_target else num_steps - 1,
        initial_dist,
        prior_to_target,
    )

    # forward: num_steps + 1 + 1
    # backward: num_steps + 1

    if prior_to_target:
        trajectories_length = (~terminals_mask).sum(axis=1)
        terminal_xs = trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ]

        # We need to add fwd/bwd_log_prob for the first continuous xs
        # fmt: off
        init_xs = jax.lax.stop_gradient(input_states)
        init_fwd_log_probs = initial_dist.log_prob(init_xs)
        bwd_clf_logits, *_ = model_state.apply_fn(params, init_xs, predict_bwd=True)

        fwd_log_probs = jnp.concatenate(
            [init_fwd_log_probs[:, None], fwd_log_probs], axis=1
        )
        bwd_log_probs = jnp.concatenate(
            [nn.log_sigmoid(bwd_clf_logits)[:, None], bwd_log_probs], axis=1
        )
        log_fs = jnp.concatenate(
            [
                jnp.zeros_like(init_fwd_log_probs)[:, None],  # set to logZ later
                log_fs,
                jnp.zeros_like(init_fwd_log_probs)[:, None], # reward is already in bwd_log_probs
            ],
            axis=1,
        )
    else:
        # fmt: off
        terminals_mask = jnp.concatenate([jnp.zeros((terminals_mask.shape[0], 1), dtype=terminals_mask.dtype), terminals_mask[:, :-1]], axis=1)

        trajectories_length = (~terminals_mask).sum(axis=1)

        trajectories = trajectories[:, ::-1]
        terminals_mask = terminals_mask[:, ::-1]
        fwd_log_probs = fwd_log_probs[:, ::-1]
        bwd_log_probs = bwd_log_probs[:, ::-1]
        log_fs = log_fs[:, ::-1]

        indices = (~terminals_mask).argsort(axis=1, descending=True, stable=True)
        trajectories = jnp.take_along_axis(trajectories, indices[:,:,None], axis=1)
        fwd_log_probs = jnp.take_along_axis(fwd_log_probs, indices, axis=1)
        bwd_log_probs = jnp.take_along_axis(bwd_log_probs, indices, axis=1)
        log_fs = jnp.take_along_axis(log_fs, indices, axis=1)

        if log_rewards is None:
            log_rewards = target.log_prob(terminal_xs)

        log_fs = jnp.concatenate(
            [
                log_fs,
                jnp.zeros_like(log_rewards)[:, None],
            ],
            axis=1,
        )

        terminal_xs = jax.lax.stop_gradient(terminal_xs)
        *_, terminal_log_fs = model_state.apply_fn(params, terminal_xs, log_rewards, predict_fwd=True)
        log_fs = log_fs.at[jnp.arange(log_fs.shape[0]), trajectories_length].set(
            terminal_log_fs
        )
        trajectories = jnp.concatenate(
            [
                trajectories[:, 1:],
                jnp.zeros_like(trajectories[:, 0:1]),
            ],
            axis=1,
        )
        trajectories = trajectories.at[jnp.arange(trajectories.shape[0]), trajectories_length - 1].set(
            terminal_xs
        )

    logZ = params["params"]["logZ"]
    log_fs = log_fs.at[:, 0].set(logZ)

    if log_rewards is None:
        log_rewards = target.log_prob(terminal_xs)

    # jax.debug.print(
    #     f"Max Length: {fwd_log_probs.shape}, {bwd_log_probs.shape}, {bwd_log_probs.shape}"
    # )
    # jax.debug.print(f"Mean Length: {trajectories_length.mean()}")

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    return (
        trajectories,
        log_pfs_over_pbs.sum(1),  # running costs
        jnp.zeros_like(log_rewards),  # stochastic costs
        -log_rewards,  # terminal costs
        trajectories_length,
        log_pfs_over_pbs,
        log_fs,
    )


def get_step_fn(aux_tuple, target, name):
    def compute_log_reward_and_langevin(s):
        return jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))

    (gamma,) = aux_tuple

    def ula_step(s, key_gen):
        langevin = jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
        fwd_mean = s + langevin * gamma
        fwd_scale = jnp.sqrt(2 * gamma)
        return sample_kernel(key_gen, fwd_mean, fwd_scale)

    def mala_step(s, key_gen):
        log_reward, langevin = compute_log_reward_and_langevin(s)
        fwd_mean = s + langevin * gamma
        fwd_scale = jnp.sqrt(2 * gamma)
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)

        log_reward_next, langevin_next = compute_log_reward_and_langevin(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        bwd_mean = s_next + langevin_next * gamma
        bwd_scale = jnp.sqrt(2 * gamma)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale)

        log_accept = log_reward_next + bwd_log_prob - log_reward - fwd_log_prob
        key, key_gen = jax.random.split(key_gen)
        accept_mask = jnp.log(jax.random.uniform(key)) < log_accept

        new_state = jnp.where(accept_mask, s_next, s)
        return new_state, key_gen

    return {"ula": ula_step, "mala": mala_step}[name]


def per_sample_rnd_cont(
    key,
    model_state,
    params,
    input_state: Array,
    step_fn,
    num_steps,
):
    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        next_state = step_fn(s, key_gen)
        per_step_output = (s,)
        return next_state, per_step_output

    init_x = input_state
    aux = (init_x, key)
    aux, per_step_output = jax.lax.scan(
        simulate_prior_to_target, aux, jnp.arange(num_steps)
    )
    terminal_x, _ = aux
    (trajectory,) = per_step_output
    return terminal_x, trajectory


def rnd_cont(
    key_gen,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    step_name,
    prior_to_target=True,
    initial_dist: distrax.Distribution | None = None,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
):
    key, key_gen = jax.random.split(key_gen)
    input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))

    step_fn = get_step_fn(aux_tuple, target, step_name)

    keys = jax.random.split(key_gen, num=batch_size)
    terminal_xs, trajectories = jax.vmap(
        per_sample_rnd_cont,
        in_axes=(0, None, None, 0, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        step_fn,
        num_steps,
    )
    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)
    log_rewards = target.log_prob(terminal_xs)

    return (
        trajectories,
        jnp.zeros_like(log_rewards),  # running costs
        jnp.zeros_like(log_rewards),  # stochastic costs
        -log_rewards,  # terminal costs
        (num_steps + 1) * jnp.ones((batch_size,), dtype=int),
        jnp.zeros_like(log_rewards),
        jnp.zeros_like(log_rewards),
    )


def loss_fn_db(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    reg_coef: float = 0.0,
    huber_delta: float | None = None,
):
    (
        trajectories,
        _,  # running costs
        _,  # stochastic costs
        terminal_costs,
        trajectories_length,
        log_pfs_over_pbs,
        log_fs,
    ) = rnd_partial(key, model_state, params)

    db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    # Only keep the db_discrepancy values corresponding to valid steps; others set to 0.
    mask = jnp.arange(db_discrepancy.shape[1])[None, :] < trajectories_length[:, None]
    db_discrepancy = db_discrepancy * mask

    if huber_delta is not None:
        db_losses = jnp.where(
            jnp.abs(db_discrepancy) <= huber_delta,
            jnp.square(db_discrepancy),
            huber_delta * jnp.abs(db_discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        db_losses = jnp.square(db_discrepancy)

    losses = db_losses.mean(-1) + reg_coef * log_fs[:, 1:].mean(-1)

    return jnp.mean(losses), (
        trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ],  # samples
        jax.lax.stop_gradient(-log_pfs_over_pbs).sum(-1),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(db_losses),
    )


def loss_fn_subtb(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    n_chunks: int,
    reg_coef: float = 0.0,
    huber_delta: float | None = None,
):
    (
        trajectories,
        _,  # running costs
        _,  # stochastic costs
        terminal_costs,
        trajectories_length,
        log_pfs_over_pbs,
        log_fs,
    ) = rnd_partial(key, model_state, params)

    bs, T = log_pfs_over_pbs.shape
    assert T % n_chunks == 0
    chunk_size = T // n_chunks

    mask = jnp.arange(T)[None, :] < trajectories_length[:, None]
    log_pfs_over_pbs = log_pfs_over_pbs * mask

    chunk_starts = jnp.arange(0, T, chunk_size)
    chunk_ends = jnp.arange(chunk_size, T + 1, chunk_size)

    valid_chunks = chunk_starts[None, :] < trajectories_length[:, None]
    chunk_valid_ends = jnp.minimum(chunk_ends[None, :], trajectories_length[:, None])

    log_fs_chunk_ends = jnp.take_along_axis(log_fs, chunk_valid_ends, axis=1)

    subtb_discrepancy1 = (
        log_fs[:, :-1:chunk_size]
        + log_pfs_over_pbs.reshape(bs, n_chunks, -1).sum(-1)
        - log_fs_chunk_ends
    )
    subtb_discrepancy1 = subtb_discrepancy1 * valid_chunks

    log_pfs_over_pbs_cumsum = jnp.cumsum(log_pfs_over_pbs[:, ::-1], axis=-1)[:, ::-1]
    subtb_discrepancy2 = (
        log_fs[:, :-1:chunk_size]
        + log_pfs_over_pbs_cumsum[:, ::chunk_size]
        - log_fs[jnp.arange(bs), trajectories_length][:, None]
    )
    # TODO: Need weights here
    subtb_discrepancy2 = subtb_discrepancy2 * valid_chunks
    subtb_discrepancy = jnp.concatenate(
        [subtb_discrepancy1, subtb_discrepancy2], axis=1
    )

    if huber_delta is not None:
        subtb_losses = jnp.where(
            jnp.abs(subtb_discrepancy) <= huber_delta,
            jnp.square(subtb_discrepancy),
            huber_delta * jnp.abs(subtb_discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        subtb_losses = jnp.square(subtb_discrepancy)

    losses = subtb_losses.mean(-1) + reg_coef * log_fs[:, 1:].mean(-1)

    return jnp.mean(losses), (
        trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ],  # samples
        jax.lax.stop_gradient(-log_pfs_over_pbs).sum(-1),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(subtb_losses),
    )


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # Define a dummy model class where we can hardcode values for testing
    class DummyModel:
        def apply_fn(
            self,
            params,
            s,
            log_reward=None,
            lgv_term=None,
            predict_fwd=False,
            predict_bwd=False,
        ):
            # Emulate logic in non_acyclic_net.py's __call__
            # Only basic logic/hardcoded but preserves output structure
            if predict_fwd:
                # Preserve same shape as s: (batch..., dim). For 1d s (dim,) use batch_shape=().
                batch_shape = s.shape[:-1] if len(s.shape) > 1 else ()
                fwd_clf_logits = jnp.full(batch_shape, 0.0)  # mimic logits
                fwd_mean = s
                fwd_scale = jnp.full(batch_shape + (s.shape[-1],), 1.0)
                if log_reward is None:
                    log_flow = jnp.zeros(batch_shape).astype(jnp.float32)
                else:
                    log_flow = s.astype(jnp.float32).squeeze(-1)
                return fwd_clf_logits, fwd_mean, fwd_scale, log_flow
            if predict_bwd:
                batch_shape = s.shape[:-1] if len(s.shape) > 1 else ()
                bwd_clf_logits = jnp.full(batch_shape, 0.0)  # mimic other logits
                bwd_mean = s
                bwd_scale = jnp.full(batch_shape + (s.shape[-1],), 1.0)
                return bwd_clf_logits, bwd_mean, bwd_scale

    class DummyTrainState:
        def __init__(self):
            self.apply_fn = DummyModel().apply_fn
            self.params = {"params": {"logZ": -1.0}}
            self.tx = None  # optimizer dummy

    class DummyTarget:
        def log_prob(self, x):
            return -42 * jnp.ones_like(x[..., 0])

    class DummyInitialDist:
        def sample(self, seed, sample_shape):
            return jnp.ones(sample_shape, dtype=jnp.float32)[:, None]

        def log_prob(self, x):
            return -13 * jnp.ones_like(x[..., 0])

    # Use the dummy train state instead of the real one
    model_state = DummyTrainState()
    params = model_state.params
    rnd_partial = rnd_with_term(
        key,
        model_state,
        params,
        2,
        (1.0,),
        DummyTarget(),
        num_steps=5,
        prior_to_target=False,
        initial_dist=DummyInitialDist(),
        terminal_xs=jnp.array([3.0, 2.0])[:, None],
    )
