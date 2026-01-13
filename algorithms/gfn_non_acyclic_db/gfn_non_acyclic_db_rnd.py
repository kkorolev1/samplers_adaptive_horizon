from typing import Callable, Literal

import distrax
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpyro.distributions as npdist
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.gfn_non_acyclic_db.sampling_utils import binary_search_smoothing


def sample_kernel(key_gen, mean, scale):
    key, key_gen = jax.random.split(key_gen)
    eps = jnp.clip(jax.random.normal(key, shape=(mean.shape[0],)), -4.0, 4.0)
    return mean + scale * eps, key_gen


def log_prob_kernel(x, mean, scale):
    dist = npdist.Independent(npdist.Normal(loc=mean, scale=scale), 1)
    return dist.log_prob(x)


def per_sample_rnd(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    prior_to_target=True,
):
    step_size = aux_tuple

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        s = jax.lax.stop_gradient(s)

        ((fwd_clf_logits, fwd_mean, fwd_scale), _, log_f) = model_state.apply_fn(
            params,
            s,
            *jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s)),
        )
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + nn.log_sigmoid(
            -fwd_clf_logits
        )

        (_, (bwd_clf_logits, bwd_mean, bwd_scale), _) = model_state.apply_fn(
            params, s_next
        )
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
        (_, (bwd_clf_logits, bwd_mean, bwd_scale), _) = model_state.apply_fn(
            params, s_next
        )
        s, key_gen = sample_kernel(key_gen, bwd_mean, bwd_scale)
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale) + jax.nn.log_sigmoid(
            -bwd_clf_logits
        )

        ((fwd_clf_logits, fwd_mean, fwd_scale), _, log_f) = model_state.apply_fn(
            params,
            s,
            *jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s)),
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
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )

    trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob, log_f


def per_sample_rnd_eval(
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
    step_size = aux_tuple

    def simulate_prior_to_target(state, per_step_input):
        s, is_terminal, key_gen = state
        s = jax.lax.stop_gradient(s)

        def term_step(s, key_gen):
            next_state = (jnp.zeros_like(s), jnp.array(True), key_gen)
            per_step_output = (s, jnp.array(True), jnp.array(0.0), jnp.array(0.0))
            return next_state, per_step_output

        def non_term_step(s, key_gen):
            ((fwd_clf_logits, fwd_mean, fwd_scale), _, _) = model_state.apply_fn(
                params, s, lgv_term=jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
            )
            key, key_gen = jax.random.split(key_gen)
            is_terminal_next = jax.random.bernoulli(key, nn.sigmoid(fwd_clf_logits))
            s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
            s_next = jax.lax.stop_gradient(s_next)
            fwd_log_prob = log_prob_kernel(
                s_next, fwd_mean, fwd_scale
            ) + nn.log_sigmoid(-fwd_clf_logits)
            fwd_log_prob = jnp.where(
                is_terminal_next, nn.log_sigmoid(fwd_clf_logits), fwd_log_prob
            )
            (_, (bwd_clf_logits, bwd_mean, bwd_scale), _) = model_state.apply_fn(
                params, s_next
            )
            bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale) + jax.nn.log_sigmoid(
                -bwd_clf_logits
            )
            bwd_log_prob = jnp.where(is_terminal_next, target.log_prob(s), bwd_log_prob)

            next_state = (s_next, is_terminal_next, key_gen)
            per_step_output = (s, jnp.array(False), fwd_log_prob, bwd_log_prob)
            return next_state, per_step_output

        return jax.lax.cond(
            is_terminal,
            term_step,
            non_term_step,
            s,
            key_gen,
        )

    def simulate_target_to_prior(state, per_step_input):
        s_next, is_terminal_next, key_gen = state
        s_next = jax.lax.stop_gradient(s_next)

        def term_step(s_next, key_gen):
            next_state = (jnp.zeros_like(s_next), jnp.array(True), key_gen)
            per_step_output = (s_next, jnp.array(True), jnp.array(0.0), jnp.array(0.0))
            return next_state, per_step_output

        def non_term_step(s_next, key_gen):
            (_, (bwd_clf_logits, bwd_mean, bwd_scale), _) = model_state.apply_fn(
                params, s_next
            )
            key, key_gen = jax.random.split(key_gen)
            is_terminal = jax.random.bernoulli(key, nn.sigmoid(bwd_clf_logits))
            s, key_gen = sample_kernel(key_gen, bwd_mean, bwd_scale)
            s = jax.lax.stop_gradient(s)
            bwd_log_prob = log_prob_kernel(s, bwd_mean, bwd_scale) + nn.log_sigmoid(
                -bwd_clf_logits
            )
            bwd_log_prob = jnp.where(
                is_terminal, nn.log_sigmoid(bwd_clf_logits), bwd_log_prob
            )

            ((fwd_clf_logits, fwd_mean, fwd_scale), _, _) = model_state.apply_fn(
                params, s, lgv_term=jax.lax.stop_gradient(jax.grad(target.log_prob)(s))
            )

            fwd_log_prob = log_prob_kernel(
                s_next, fwd_mean, fwd_scale
            ) + jax.nn.log_sigmoid(-fwd_clf_logits)
            fwd_log_prob = jnp.where(
                is_terminal, initial_dist.log_prob(s_next), fwd_log_prob
            )

            next_state = (s, is_terminal, key_gen)
            per_step_output = (s, is_terminal, fwd_log_prob, bwd_log_prob)
            return next_state, per_step_output

        return jax.lax.cond(
            is_terminal_next,
            term_step,
            non_term_step,
            s_next,
            key_gen,
        )

    if prior_to_target:
        init_x = input_state
        aux = (init_x, jnp.array(False), key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(num_steps)
        )
        terminal_x, is_terminal_x, _ = aux
        trajectory, terminal_mask, _, _ = per_step_output

        terminal_mask = jnp.concatenate(
            [terminal_mask, jnp.expand_dims(is_terminal_x, axis=0)], axis=0
        )
        terminal_index = jnp.where(~terminal_mask, size=num_steps)[0].max()
        full_traj = jnp.concatenate(
            [trajectory, jnp.expand_dims(terminal_x, axis=0)], axis=0
        )
        terminal_x = full_traj[terminal_index]
    else:
        terminal_x = input_state
        aux = (terminal_x, jnp.array(False), key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )

    trajectory, terminal_mask, fwd_log_prob, bwd_log_prob = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob


def rnd(
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
        if initial_dist is None:  # pinned_brownian
            input_states = jnp.zeros((batch_size, target.dim))
        else:
            key, key_gen = jax.random.split(key_gen)
            input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    terminal_xs, trajectories, fwd_log_probs, bwd_log_probs, log_fs = jax.vmap(
        per_sample_rnd,
        in_axes=(0, None, None, 0, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps,
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
    (_, _, terminal_log_fs) = jax.vmap(model_state.apply_fn, in_axes=(None, 0, 0))(
        params, terminal_xs, log_rewards
    )
    # We need to calculate bwd_log_prob for the first continuous xs
    init_xs = jax.lax.stop_gradient(trajectories[:, 0])
    (_, (bwd_clf_logits, *_), _) = jax.vmap(model_state.apply_fn, in_axes=(None, 0))(
        params, init_xs
    )
    fwd_log_probs = jnp.concatenate(
        [init_fwd_log_probs[:, None], fwd_log_probs], axis=1
    )
    bwd_log_probs = jnp.concatenate(
        [nn.log_sigmoid(bwd_clf_logits)[:, None], bwd_log_probs], axis=1
    )
    log_fs = jnp.concatenate(
        [
            jnp.zeros_like(terminal_log_fs)[:, None],  # set to logZ in loss
            log_fs,
            terminal_log_fs[:, None],
        ],
        axis=1,
    )
    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    return (
        trajectories[:, -1],
        log_pfs_over_pbs.sum(1),
        jnp.zeros_like(log_rewards),
        -log_rewards,
        trajectories,
        log_pfs_over_pbs,  # log_pfs - log_pbs
        log_fs,
    )


def rnd_eval(
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
        if initial_dist is None:  # pinned_brownian
            input_states = jnp.zeros((batch_size, target.dim))
        else:
            key, key_gen = jax.random.split(key_gen)
            input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    terminal_xs, trajectories, fwd_log_probs, bwd_log_probs = jax.vmap(
        per_sample_rnd_eval,
        in_axes=(0, None, None, 0, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps,
        initial_dist,
        prior_to_target,
    )

    if not prior_to_target:
        trajectories = trajectories[:, ::-1]
        fwd_log_probs = fwd_log_probs[:, ::-1]
        bwd_log_probs = bwd_log_probs[:, ::-1]

    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)

    if log_rewards is None:
        log_rewards = target.log_prob(terminal_xs)

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    return (
        trajectories[:, -1],
        log_pfs_over_pbs.sum(1),
        jnp.zeros_like(log_rewards),
        -log_rewards,
        trajectories,
        log_pfs_over_pbs,  # log_pfs - log_pbs
    )


def loss_fn(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
):
    # samples, running_costs, stochastic_costs, terminal_costs,...
    (
        _,
        _,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        log_fs,
    ) = rnd_partial(key, model_state, params)

    logZ = params["params"]["logZ"]
    log_fs = log_fs.at[:, 0].set(logZ)
    db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]

    if huber_delta is not None:
        db_losses = jnp.where(
            jnp.abs(db_discrepancy) <= huber_delta,
            jnp.square(db_discrepancy),
            huber_delta * jnp.abs(db_discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        db_losses = jnp.square(db_discrepancy)

    return jnp.mean(db_losses.mean(-1)), (
        trajectories,
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(db_losses),
    )
