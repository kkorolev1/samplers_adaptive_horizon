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


def per_sample_rnd_train(
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

    def model_forward(s, t, log_reward, langevin):
        return model_state.apply_fn(
            params, s, t, log_reward, langevin, predict_fwd=True
        )

    def model_backward(s_next, t_next):
        return model_state.apply_fn(params, s_next, t_next, predict_fwd=False)

    def compute_log_reward_and_langevin(s):
        return jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps
        s = jax.lax.stop_gradient(s)

        log_reward, langevin = compute_log_reward_and_langevin(s)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s, t * jnp.ones(1), log_reward, langevin
        )
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + nn.log_sigmoid(
            -fwd_clf_logits
        )

        bwd_mean, bwd_scale = model_backward(s_next, t_next)
        bwd_log_prob = jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(*args),
            operand=(s, bwd_mean, bwd_scale),
        )

        # Return next state and per-step output
        next_state = (s_next, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, fwd_clf_logits, log_f)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        s_next, key_gen = state
        step = per_step_input
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps
        s_next = jax.lax.stop_gradient(s_next)
        bwd_mean, bwd_scale = model_backward(s_next, t_next)
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
        log_reward, langevin = compute_log_reward_and_langevin(s)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s, t, log_reward, langevin
        )
        fwd_log_prob = log_prob_kernel(
            s_next, fwd_mean, fwd_scale
        ) + jax.nn.log_sigmoid(-fwd_clf_logits)

        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, fwd_clf_logits, log_f)
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
    trajectory, fwd_log_prob, bwd_log_prob, fwd_clf_logits, log_f = per_step_output
    return (
        terminal_x,
        trajectory,
        fwd_log_prob,
        bwd_log_prob,
        fwd_clf_logits,
        log_f,
    )


def rnd_train(
    key_gen,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    prior_to_target=True,
    initial_dist=None,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
):
    if prior_to_target:
        input_states = jnp.zeros((batch_size, target.dim))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    (
        terminal_xs,
        trajectories,
        fwd_log_probs,
        bwd_log_probs,
        fwd_clf_logits,
        log_fs,
    ) = jax.vmap(
        per_sample_rnd_train,
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
        fwd_clf_logits = fwd_clf_logits[:, ::-1]
        log_fs = log_fs[:, ::-1]

    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)

    if log_rewards is None:
        log_rewards = target.log_prob(terminal_xs)

    # We need to calculate log flow and fwd clf logits for the last continuous xs
    terminal_xs = jax.lax.stop_gradient(terminal_xs)
    terminal_fwd_clf_logits, _, _, terminal_log_fs = model_state.apply_fn(
        params, terminal_xs, log_rewards, predict_fwd=True
    )
    fwd_clf_logits = jnp.concatenate(
        [fwd_clf_logits, terminal_fwd_clf_logits[:, None]], axis=1
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
        log_rewards,
        log_pfs_over_pbs,
        fwd_clf_logits,
        log_fs,
    )


def loss_fn_prefix_tb(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    reg_coef: float = 0.0,
    huber_delta: float | None = None,
    use_weights: bool = True,
):
    (trajectories, log_rewards, log_pfs_over_pbs, fwd_clf_logits, log_fs) = rnd_partial(
        key, model_state, params
    )

    fwd_clf_log_probs = jax.nn.log_sigmoid(fwd_clf_logits)
    logZ = params["params"]["logZ"]
    discrepancy = logZ + jnp.cumsum(log_pfs_over_pbs, axis=1) - log_fs[:, 1:]

    if use_weights:
        log_weights = (
            jnp.cumsum(
                jnp.concatenate(
                    [
                        jnp.zeros((fwd_clf_logits.shape[0], 1)),
                        jax.nn.log_sigmoid(-fwd_clf_logits)[:, :-1],
                    ],
                    axis=1,
                ),
                axis=1,
            )
            + fwd_clf_log_probs
        )
        log_weights = jax.lax.stop_gradient(log_weights)
        weights = jnp.exp(
            log_weights
            - jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
        )
    else:
        weights = jnp.ones((fwd_clf_logits.shape[0], 1)) / fwd_clf_logits.shape[1]

    if huber_delta is not None:
        tb_losses = jnp.where(
            jnp.abs(discrepancy) <= huber_delta,
            jnp.square(discrepancy),
            huber_delta * jnp.abs(discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        tb_losses = jnp.square(discrepancy)

    tb_losses = tb_losses * weights
    losses = tb_losses.sum(-1) + reg_coef * (jnp.exp(log_fs[:, 1:]) * weights).sum(-1)

    return jnp.mean(losses), (
        trajectories[:, -1],
        jax.lax.stop_gradient(-log_pfs_over_pbs).sum(-1),
        log_rewards,
        jax.lax.stop_gradient(tb_losses),
        weights,
    )
