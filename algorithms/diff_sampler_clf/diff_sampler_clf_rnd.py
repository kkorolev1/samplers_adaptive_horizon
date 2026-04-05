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

    def model_forward(s, t, log_reward, langevin):
        return model_state.apply_fn(
            params, s, t, log_reward, langevin, predict_fwd=True
        )

    def model_backward(s_next, t_next):
        return model_state.apply_fn(params, s_next, t_next, predict_fwd=False)

    def compute_log_reward_and_langevin(s, t):
        if use_lp:
            log_reward, langevin = jax.lax.stop_gradient(
                jax.value_and_grad(target.log_prob)(s)
            )
        else:
            log_reward = jax.lax.stop_gradient(target.log_prob(s))
            langevin = None
        # TODO: Replace to any step dist
        log_reward = log_reward - jnp.log(num_steps)
        return log_reward, langevin

    def simulate_prior_to_target(state, per_step_input):
        s, key_gen = state
        step = per_step_input
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps
        s = jax.lax.stop_gradient(s)

        log_reward, langevin = compute_log_reward_and_langevin(s, t)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s, t * jnp.ones(1), log_reward, langevin
        )
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        # we don't terminate at step 0
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda logits: nn.log_sigmoid(-logits),
            operand=fwd_clf_logits,
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
        log_reward, langevin = compute_log_reward_and_langevin(s, t)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, log_f = model_forward(
            s, t, log_reward, langevin
        )
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda logits: nn.log_sigmoid(-logits),
            operand=fwd_clf_logits,
        )

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
        fwd_clf_logits = fwd_clf_logits[:, ::-1]
        log_fs = log_fs[:, ::-1]

    trajectories = jnp.concatenate([trajectories, terminal_xs[:, None]], axis=1)

    if log_rewards is None:
        # TODO: Replace to any step dist
        log_rewards = target.log_prob(terminal_xs) - jnp.log(num_steps)

    # We need to calculate log flow and fwd clf logits for the last continuous xs
    terminal_xs = jax.lax.stop_gradient(terminal_xs)
    terminal_fwd_clf_logits, _, _, terminal_log_fs = model_state.apply_fn(
        params,
        terminal_xs,
        num_steps * jnp.ones((terminal_xs.shape[0], 1), dtype=int),
        log_rewards,
        predict_fwd=True,
    )
    fwd_clf_logits = jnp.concatenate(
        [fwd_clf_logits, terminal_fwd_clf_logits[:, None]], axis=1
    )
    logZ = params["params"]["logZ"]
    log_fs = jnp.concatenate(
        [
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
                        jax.nn.log_sigmoid(-fwd_clf_logits)[:, 1:-1],
                    ],
                    axis=1,
                ),
                axis=1,
            )
            + fwd_clf_log_probs[:, 1:]
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


def per_sample_rnd_eval(
    key,
    model_state,
    params,
    input_state: Array,
    input_step: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    initial_dist=None,
    prior_to_target=True,
):
    (logr_clip,) = aux_tuple

    def model_forward(s, t, langevin, force_stop=False):
        return model_state.apply_fn(
            params,
            s,
            t,
            None,
            langevin,
            predict_fwd=True,
            force_stop=force_stop,
        )

    def model_backward(s_next, t_next, force_stop=False):
        return model_state.apply_fn(
            params,
            s_next,
            t_next,
            predict_fwd=False,
            force_stop=force_stop,
        )

    def compute_log_reward_and_langevin(s, t):
        if use_lp:
            log_reward, langevin = jax.lax.stop_gradient(
                jax.value_and_grad(target.log_prob)(s)
            )
        else:
            log_reward = jax.lax.stop_gradient(target.log_prob(s))
            langevin = None
        # TODO: Replace to any step dist
        log_reward = log_reward - jnp.log(num_steps)
        return log_reward, langevin

    def cond_fun(carry):
        state, _ = carry
        s, is_terminal, key_gen, step = state
        return (jnp.any(~is_terminal)) & (step < num_steps)

    def simulate_prior_to_target(carry, force_stop=False):
        state, state_hist = carry
        s, is_terminal, key_gen, step = state
        trajectories, terminals_mask, fwd_log_probs, bwd_log_probs = state_hist
        _step = step.astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s = jax.lax.stop_gradient(s)
        log_reward, langevin = compute_log_reward_and_langevin(s, t)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, _ = model_forward(
            s,
            t,
            langevin,
            force_stop=force_stop,
        )
        key, key_gen = jax.random.split(key_gen)
        is_terminal_next = is_terminal | (
            jax.random.bernoulli(key, nn.sigmoid(fwd_clf_logits)) & (step > 0)
        )
        s_next, key_gen = sample_kernel(key_gen, fwd_mean, fwd_scale)
        s_next = jax.lax.stop_gradient(s_next)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda logits: nn.log_sigmoid(-logits),
            operand=fwd_clf_logits,
        )
        fwd_log_prob = jnp.where(
            is_terminal_next, nn.log_sigmoid(fwd_clf_logits), fwd_log_prob
        )
        fwd_log_prob = jnp.where(
            is_terminal, jnp.zeros_like(fwd_clf_logits), fwd_log_prob
        )
        bwd_mean, bwd_scale = model_backward(s_next, t_next)
        bwd_log_prob = jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(*args),
            operand=(s, bwd_mean, bwd_scale),
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

        # Reverse step to get t in target to prior
        _step = (num_steps - 1 - step).astype(jnp.float32)
        t = _step / num_steps
        t_next = (_step + 1) / num_steps

        s_next = jax.lax.stop_gradient(s_next)
        bwd_mean, bwd_scale = model_backward(
            s_next,
            t_next,
            force_stop=force_stop,
        )
        # we don't terminate during backward
        is_terminal = is_terminal_next
        s, key_gen = jax.lax.cond(
            _step.astype(int) == 0,
            lambda _: (jnp.zeros_like(s_next), key_gen),
            lambda args: sample_kernel(*args),
            operand=(key_gen, bwd_mean, bwd_scale),
        )
        s = jax.lax.stop_gradient(s)
        bwd_log_prob = jax.lax.cond(
            _step.astype(int) == 0,
            lambda _: jnp.array(0.0),
            lambda args: log_prob_kernel(*args),
            operand=(s, bwd_mean, bwd_scale),
        )
        log_reward, langevin = compute_log_reward_and_langevin(s, t)
        log_reward = clip_log_reward(log_reward, clip_value=logr_clip)
        fwd_clf_logits, fwd_mean, fwd_scale, _ = model_forward(s, t, langevin)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale) + jax.lax.cond(
            step == 0,
            lambda _: jnp.array(0.0),
            lambda logits: nn.log_sigmoid(-logits),
            operand=fwd_clf_logits,
        )

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
    state_init = (input_state, jnp.array(False), key, input_step)
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
    initial_dist=None,
    terminal_xs: Array | None = None,
    log_rewards: Array | None = None,
):
    if prior_to_target:
        input_states = jnp.zeros((batch_size, target.dim))
        input_steps = jnp.zeros((batch_size,), dtype=int)
    else:
        input_states = terminal_xs
        # TODO: Replace to any step dist
        key_gen, key = jax.random.split(key_gen)
        input_steps = jax.random.randint(key, (batch_size,), 1, num_steps, dtype=int)

    keys = jax.random.split(key_gen, num=batch_size)
    trajectories, terminals_mask, fwd_log_probs, bwd_log_probs = jax.vmap(
        per_sample_rnd_eval,
        in_axes=(0, None, None, 0, 0, None, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        input_steps,
        aux_tuple,
        target,
        num_steps,
        use_lp,
        initial_dist,
        prior_to_target,
    )

    running_costs = jnp.sum(fwd_log_probs - bwd_log_probs, axis=1)

    def compute_log_reward(s, t):
        # TODO: Replace to any step dist
        return target.log_prob(s) - jnp.log(num_steps)

    if prior_to_target:
        trajectories_length = (~terminals_mask).sum(axis=1)
        terminal_xs = trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ]
        log_rewards = compute_log_reward(terminal_xs, trajectories_length - 1)
    else:
        # fmt: off
        
        # account +1 for the terminal from which we started
        trajectories_length = (~terminals_mask).sum(axis=1) + 1
        trajectories = trajectories[:, ::-1]
        terminals_mask = terminals_mask[:, ::-1]

        indices = (~terminals_mask).argsort(axis=1, descending=True, stable=True)
        trajectories = jnp.take_along_axis(trajectories, indices[:,:,None], axis=1)

        log_rewards = compute_log_reward(terminal_xs, trajectories_length - 1)

        terminal_xs = jax.lax.stop_gradient(terminal_xs)
        fwd_clf_logits, *_ = model_state.apply_fn(params, terminal_xs, input_steps[:, None], predict_fwd=True)
        running_costs = running_costs + nn.log_sigmoid(fwd_clf_logits) - log_rewards
        trajectories = trajectories.at[jnp.arange(trajectories.shape[0]), trajectories_length - 1].set(
            terminal_xs
        )

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
