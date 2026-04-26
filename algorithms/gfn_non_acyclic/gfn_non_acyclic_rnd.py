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
    use_lp,
    prior_to_target=True,
):
    (logr_clip,) = aux_tuple

    def model_forward(s, log_reward, langevin):
        return model_state.apply_fn(params, s, log_reward, langevin, predict_fwd=True)

    def model_backward(s_next):
        return model_state.apply_fn(params, s_next, predict_fwd=False)

    def compute_log_reward_and_langevin(s):
        if use_lp:
            return jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
        return jax.lax.stop_gradient(target.log_prob(s)), None

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
        per_step_output = (s, fwd_log_prob, bwd_log_prob, fwd_clf_logits, log_f)
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
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
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
    use_lp,
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
    (
        terminal_xs,
        trajectories,
        fwd_log_probs,
        bwd_log_probs,
        fwd_clf_logits,
        log_fs,
    ) = jax.vmap(
        per_sample_rnd_train,
        in_axes=(0, None, None, 0, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps - 1,
        use_lp,
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

    if initial_dist is None:  # pinned_brownian
        init_fwd_log_probs = jnp.zeros(batch_size)
    else:
        init_fwd_log_probs = initial_dist.log_prob(trajectories[:, 0])

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

    # We need to calculate forward and backward log probs for the first continuous xs
    init_xs = jax.lax.stop_gradient(trajectories[:, 0])
    bwd_clf_logits, *_ = model_state.apply_fn(params, init_xs, predict_fwd=False)

    fwd_log_probs = jnp.concatenate(
        [init_fwd_log_probs[:, None], fwd_log_probs], axis=1
    )
    bwd_log_probs = jnp.concatenate(
        [nn.log_sigmoid(bwd_clf_logits)[:, None], bwd_log_probs], axis=1
    )

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
    only_clf_reg: bool = False,
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
    reg_term = jnp.exp(-fwd_clf_log_probs) if only_clf_reg else jnp.exp(log_fs[:, 1:])

    losses = tb_losses.sum(-1) + reg_coef * (reg_term * weights).sum(-1)

    return jnp.mean(losses), (
        trajectories[:, -1],
        jax.lax.stop_gradient(-log_pfs_over_pbs).sum(-1),
        log_rewards,
        jax.lax.stop_gradient(tb_losses),
        weights,
    )


def loss_fn_db(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    reg_coef: float = 0.0,
    huber_delta: float | None = None,
):
    (trajectories, log_rewards, log_pfs_over_pbs, fwd_clf_logits, log_fs) = rnd_partial(
        key, model_state, params
    )

    db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    db_discrepancy = db_discrepancy

    if huber_delta is not None:
        db_losses = jnp.where(
            jnp.abs(db_discrepancy) <= huber_delta,
            jnp.square(db_discrepancy),
            huber_delta * jnp.abs(db_discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        db_losses = jnp.square(db_discrepancy)

    losses = db_losses.mean(-1) + reg_coef * jnp.exp(log_fs[:, 1:]).mean(-1)

    return jnp.mean(losses), (
        trajectories[:, -1],
        jax.lax.stop_gradient(-log_pfs_over_pbs).sum(-1),
        log_rewards,
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
    (trajectories, log_rewards, log_pfs_over_pbs, fwd_clf_logits, log_fs) = rnd_partial(
        key, model_state, params
    )

    bs, T = log_pfs_over_pbs.shape
    assert T % n_chunks == 0
    chunk_size = T // n_chunks

    # db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    # subtb_discrepancy1 = db_discrepancy.reshape(bs, n_chunks, -1).sum(-1)
    # The below is equivalent to the above lines but avoids numerical instability.
    subtb_discrepancy1 = (
        log_fs[:, :-1:chunk_size]
        + log_pfs_over_pbs.reshape(bs, n_chunks, -1).sum(-1)
        - log_fs[:, chunk_size::chunk_size]
    )

    log_pfs_over_pbs_cumsum = jnp.cumsum(log_pfs_over_pbs[:, ::-1], axis=-1)[:, ::-1]
    subtb_discrepancy2 = (
        log_fs[:, :-1:chunk_size]
        + log_pfs_over_pbs_cumsum[:, ::chunk_size]
        - log_fs[:, [-1]]
    ) / jnp.arange(1, n_chunks + 1)[None, ::-1]

    # subtb_discrepancy = subtb_discrepancy1
    # subtb_discrepancy = subtb_discrepancy2
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

    losses = subtb_losses.mean(-1) + reg_coef * jnp.exp(log_fs[:, 1:]).mean(-1)

    return jnp.mean(losses), (
        trajectories[:, -1],
        jax.lax.stop_gradient(-log_pfs_over_pbs).sum(-1),
        log_rewards,
        jax.lax.stop_gradient(subtb_losses),
    )


def per_sample_rnd_eval(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    initial_dist,
    prior_to_target=True,
):
    (logr_clip,) = aux_tuple

    def model_forward(s, langevin, force_stop=False):
        return model_state.apply_fn(
            params,
            s,
            None,
            langevin,
            predict_fwd=True,
            force_stop=force_stop,
        )

    def model_backward(s_next, force_stop=False):
        return model_state.apply_fn(
            params,
            s_next,
            predict_fwd=False,
            force_stop=force_stop,
        )

    def compute_log_reward_and_langevin(s):
        if use_lp:
            return jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
        return jax.lax.stop_gradient(target.log_prob(s)), None

    def cond_fun(carry):
        state, _ = carry
        s, is_terminal, key_gen, step = state
        return (jnp.any(~is_terminal)) & (step < num_steps)

    def simulate_prior_to_target(carry, force_stop=False):
        state, state_hist = carry
        s, is_terminal, key_gen, step = state
        trajectories, terminals_mask, fwd_log_probs, bwd_log_probs = state_hist

        s = jax.lax.stop_gradient(s)
        log_reward, langevin = compute_log_reward_and_langevin(s)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, _ = model_forward(
            s,
            langevin,
            force_stop=force_stop,
        )
        key, key_gen = jax.random.split(key_gen)
        is_terminal_next = is_terminal | jax.random.bernoulli(
            key, nn.sigmoid(fwd_clf_logits)
        )
        if force_stop:
            is_terminal_next = jnp.array(True)
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

        s_next = jnp.where(is_terminal_next, jnp.zeros_like(s_next), s_next)

        trajectories = trajectories.at[step].set(s)
        terminals_mask = terminals_mask.at[step].set(is_terminal)
        fwd_log_probs = fwd_log_probs.at[step].set(fwd_log_prob)
        bwd_log_probs = bwd_log_probs.at[step].set(bwd_log_prob)

        state_next = (s_next, is_terminal_next, key_gen, step + 1)
        state_hist = (
            trajectories,
            terminals_mask,
            fwd_log_probs,
            bwd_log_probs,
        )
        return (state_next, state_hist)

    def simulate_target_to_prior(carry, force_stop=False):
        state, state_hist = carry
        s_next, is_terminal_next, key_gen, step = state
        trajectories, terminals_mask, fwd_log_probs, bwd_log_probs = state_hist
        s_next = jax.lax.stop_gradient(s_next)
        bwd_clf_logits, bwd_mean, bwd_scale = model_backward(
            s_next,
            force_stop=force_stop,
        )
        key, key_gen = jax.random.split(key_gen)
        is_terminal = is_terminal_next | jax.random.bernoulli(
            key, nn.sigmoid(bwd_clf_logits)
        )
        if force_stop:
            is_terminal = jnp.array(True)
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
        fwd_clf_logits, fwd_mean, fwd_scale, _ = model_forward(s, langevin)

        fwd_log_prob = log_prob_kernel(
            s_next, fwd_mean, fwd_scale
        ) + jax.nn.log_sigmoid(-fwd_clf_logits)
        fwd_log_prob = jnp.where(
            is_terminal, initial_dist.log_prob(s_next), fwd_log_prob
        )
        fwd_log_prob = jnp.where(
            is_terminal_next, jnp.zeros_like(fwd_log_prob), fwd_log_prob
        )

        s = jnp.where(is_terminal, jnp.zeros_like(s), s)

        trajectories = trajectories.at[step].set(s)
        terminals_mask = terminals_mask.at[step].set(is_terminal)
        fwd_log_probs = fwd_log_probs.at[step].set(fwd_log_prob)
        bwd_log_probs = bwd_log_probs.at[step].set(bwd_log_prob)

        state_next = (s, is_terminal, key_gen, step + 1)
        state_hist = (
            trajectories,
            terminals_mask,
            fwd_log_probs,
            bwd_log_probs,
        )
        return (state_next, state_hist)

    d = input_state.shape[-1]
    trajectories = jnp.zeros((num_steps + 1, d))
    terminals_mask = jnp.ones((num_steps + 1,), dtype=bool)
    fwd_log_probs = jnp.zeros((num_steps + 1,))
    bwd_log_probs = jnp.zeros((num_steps + 1,))
    state_hist = (trajectories, terminals_mask, fwd_log_probs, bwd_log_probs)
    state_init = (input_state, jnp.array(False), key, 0)
    carry = (state_init, state_hist)

    if prior_to_target:
        carry = equinox.internal.while_loop(
            cond_fun,
            simulate_prior_to_target,
            carry,
            max_steps=num_steps,
            kind="checkpointed",
        )
        carry = simulate_prior_to_target(carry, force_stop=True)
    else:
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


def rnd_eval(
    key_gen,
    model_state,
    params,
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
        key, key_gen = jax.random.split(key_gen)
        input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))
    else:
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    trajectories, terminals_mask, fwd_log_probs, bwd_log_probs = jax.vmap(
        per_sample_rnd_eval,
        in_axes=(0, None, None, 0, None, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps - 1,
        use_lp,
        initial_dist,
        prior_to_target,
    )

    running_costs = jnp.sum(fwd_log_probs - bwd_log_probs, axis=1)

    if prior_to_target:
        trajectories_length = (~terminals_mask).sum(axis=1)
        terminal_xs = trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ]
        # We need to add fwd/bwd_log_prob for the first continuous xs
        # fmt: off
        init_xs = jax.lax.stop_gradient(input_states)
        init_fwd_log_probs = initial_dist.log_prob(init_xs)
        bwd_clf_logits, *_ = model_state.apply_fn(params, init_xs, predict_fwd=False)

        running_costs = running_costs + init_fwd_log_probs - nn.log_sigmoid(bwd_clf_logits)
    else:
        # fmt: off
        
        # account +1 for the terminal from which we started
        trajectories_length = (~terminals_mask).sum(axis=1) + 1
        trajectories = trajectories[:, ::-1]
        terminals_mask = terminals_mask[:, ::-1]

        indices = (~terminals_mask).argsort(axis=1, descending=True, stable=True)
        trajectories = jnp.take_along_axis(trajectories, indices[:,:,None], axis=1)

        if log_rewards is None:
            log_rewards = target.log_prob(terminal_xs)

        terminal_xs = jax.lax.stop_gradient(terminal_xs)
        fwd_clf_logits, *_ = model_state.apply_fn(params, terminal_xs, predict_fwd=True)
        running_costs = running_costs + nn.log_sigmoid(fwd_clf_logits) - log_rewards
        trajectories = trajectories.at[jnp.arange(trajectories.shape[0]), trajectories_length - 1].set(
            terminal_xs
        )

    if log_rewards is None:
        log_rewards = target.log_prob(terminal_xs)

    return (
        trajectories,
        running_costs,
        log_rewards,
        trajectories_length,
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


def per_sample_rnd_mcmc(
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


def rnd_mcmc(
    key_gen,
    model_state,
    params,
    batch_size,
    aux_tuple,
    target,
    num_steps,
    step_name,
    initial_dist: distrax.Distribution | None = None,
    prior_to_target: bool = True,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
    input_states: Array | None = None,
):
    if input_states is None:
        key, key_gen = jax.random.split(key_gen)
        input_states = initial_dist.sample(seed=key, sample_shape=(batch_size,))

    step_fn = get_step_fn(aux_tuple, target, step_name)

    keys = jax.random.split(key_gen, num=batch_size)
    terminal_xs, trajectories = jax.vmap(
        per_sample_rnd_mcmc,
        in_axes=(0, None, None, 0, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        step_fn,
        num_steps - 1,
    )
    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)
    log_rewards = target.log_prob(terminal_xs)

    return (
        trajectories,
        jnp.zeros_like(log_rewards),
        log_rewards,
        num_steps * jnp.ones((trajectories.shape[0],), dtype=int),
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
            else:
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
    rnd_partial = rnd_eval(
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
