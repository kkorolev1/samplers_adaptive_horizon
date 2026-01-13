from typing import Callable, Literal

import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from algorithms.common.types import Array, RandomKey, ModelParams
from algorithms.gfn_tb.gfn_tb_rnd import sample_kernel, log_prob_kernel


def get_beta_fn(
    params, beta_schedule: Literal["learnt", "linear", "cosine"], num_steps, lambda_fn=None
):
    if beta_schedule == "learnt":
        b = jax.nn.softplus(params["params"]["betas"])
        b = jnp.cumsum(b) / jnp.sum(b)
        b = jnp.concatenate((jnp.array([0]), b))
        beta_fn = lambda step: b[step]
    elif beta_schedule == "cosine":
        assert lambda_fn is not None
        beta_fn = lambda step: (1 - lambda_fn(step))
    elif beta_schedule == "linear":
        beta_fn = lambda step: step / num_steps
    else:
        raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented.")
    return beta_fn


def get_flow_bias(weight, ref_log_prob, log_prob):
    return (1 - weight) * ref_log_prob + weight * log_prob


def ref_log_prob_pinned_brownian(s, t, sigma_t):
    ref_log_var = jnp.log((sigma_t**2) * jnp.array(t))
    # CAUTION: Note that this variance is only correct when the noise schedule is constant.
    # so, if you want to use a non-constant noise schedule, you need to modify this.
    # e.g., by keeping track of the cumulative variance through the time steps.

    log_p_ref = -0.5 * (
        jnp.log(2 * jnp.pi) + ref_log_var + jnp.exp(-ref_log_var) * jnp.square(s)
    ).sum(-1)
    return log_p_ref


def per_sample_rnd_pinned_brownian(
    key,
    model_state,
    params: ModelParams,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    partial_energy,
    beta_schedule: Literal["learnt", "linear", "cosine"],
    prior_to_target=True,
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

        # Compute forward SDE components
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

        # Compute importance weight increment
        next_state = (s_next, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
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

        fwd_mean = s + model_output * dt
        fwd_scale = sigma_t * jnp.sqrt(dt)
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Compute importance weight increment
        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    if prior_to_target:
        init_x = input_state
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(num_steps))
        terminal_x, _ = aux
    else:
        terminal_x = input_state
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )

    trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob, log_f


def per_sample_rnd_ou(
    key,
    model_state,
    params,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    partial_energy,
    beta_schedule: Literal["learnt", "linear", "cosine"],
    prior_to_target=True,
):
    raise NotImplementedError("OU reference process not implemented yet.")


def per_sample_rnd_ou_dds(
    key,
    model_state,
    params: ModelParams,
    input_state: Array,
    aux_tuple,
    target,
    num_steps,
    use_lp,
    partial_energy,
    beta_schedule: Literal["learnt", "linear", "cosine"],
    prior_to_target=True,
):
    init_std, init_log_prob, alpha_fn, lambda_fn = aux_tuple
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
            log_f_bias = get_flow_bias(beta_fn(step), init_log_prob(s), log_prob)

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
        log_prob = jnp.array(0.0)
        langevin = jnp.zeros(s.shape[0])
        if use_lp:
            log_prob, langevin = jax.lax.stop_gradient(jax.value_and_grad(target.log_prob)(s))
        elif partial_energy:
            log_prob = target.log_prob(s)

        log_f_bias = jnp.array(0.0)
        if partial_energy:
            log_f_bias = get_flow_bias(beta_fn(step), init_log_prob(s), log_prob)

        model_output, log_f = model_state.apply_fn(params, s, t * jnp.ones(1), langevin)
        log_f = log_f + log_f_bias

        fwd_mean = sqrt_1_minus_at_next * s + sqrt_at_next**2 * model_output
        fwd_scale = sqrt_at_next * init_std
        fwd_log_prob = log_prob_kernel(s_next, fwd_mean, fwd_scale)

        # Return next state and per-step output
        next_state = (s, key_gen)
        per_step_output = (s, fwd_log_prob, bwd_log_prob, log_f)
        return next_state, per_step_output

    if prior_to_target:
        init_x = input_state
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(simulate_prior_to_target, aux, jnp.arange(num_steps))
        terminal_x, _ = aux
    else:
        terminal_x = input_state
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(num_steps)[::-1]
        )
        init_x, _ = aux

    trajectory, fwd_log_prob, bwd_log_prob, log_f = per_step_output
    return terminal_x, trajectory, fwd_log_prob, bwd_log_prob, log_f


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
    partial_energy,
    beta_schedule: Literal["learnt", "linear", "cosine"],
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
        assert terminal_xs is not None
        input_states = terminal_xs

    keys = jax.random.split(key_gen, num=batch_size)
    per_sample_fn = {
        "pinned_brownian": per_sample_rnd_pinned_brownian,
        "ou": per_sample_rnd_ou,
        "ou_dds": per_sample_rnd_ou_dds,
    }[reference_process]

    terminal_xs, trajectories, fwd_log_probs, bwd_log_probs, log_fs = jax.vmap(
        per_sample_fn,
        in_axes=(0, None, None, 0, None, None, None, None, None, None, None),
    )(
        keys,
        model_state,
        params,
        input_states,
        aux_tuple,
        target,
        num_steps,
        use_lp,
        partial_energy,
        beta_schedule,
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
    log_fs = jnp.concatenate([log_fs, log_rewards[:, None]], axis=1)

    log_pfs_over_pbs = fwd_log_probs - bwd_log_probs
    return (
        trajectories[:, -1],
        log_pfs_over_pbs.sum(1) + init_fwd_log_probs,
        jnp.zeros_like(log_rewards),
        -log_rewards,
        trajectories,
        log_pfs_over_pbs,  # log_pfs - log_pbs
        log_fs,
        init_fwd_log_probs,
    )


def loss_fn_tb(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    loss_type: Literal["tb", "lv"] = "tb",
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
):
    (
        _,
        running_costs,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        _,
        _,
    ) = rnd_partial(key, model_state, params)
    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    log_ratio = running_costs - log_rewards * invtemp  # running_costs already contains initial pfs
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

    return jnp.mean(losses), (
        trajectories,
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(losses),
        jnp.zeros_like(losses),
    )


def loss_fn_subtb(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    n_chunks: int,
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
    flow_only: bool = False,  # if True, only optimize flow
):
    (
        _,
        _,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        log_fs,
        init_fwd_log_probs,
    ) = rnd_partial(key, model_state, params)

    bs, T = log_pfs_over_pbs.shape
    assert T % n_chunks == 0
    chunk_size = T // n_chunks

    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    logZ = params["params"]["logZ"]
    if flow_only:
        logZ = jax.lax.stop_gradient(logZ)
        log_pfs_over_pbs = jax.lax.stop_gradient(log_pfs_over_pbs)

    log_fs = log_fs.at[:, 0].set(logZ + init_fwd_log_probs)
    log_fs = log_fs.at[:, -1].set(log_rewards * invtemp)

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
        log_fs[:, :-1:chunk_size] + log_pfs_over_pbs_cumsum[:, ::chunk_size] - log_fs[:, [-1]]
    ) / jnp.arange(1, n_chunks + 1)[None, ::-1]

    # subtb_discrepancy = subtb_discrepancy1
    # subtb_discrepancy = subtb_discrepancy2
    subtb_discrepancy = jnp.concatenate([subtb_discrepancy1, subtb_discrepancy2], axis=1)

    if huber_delta is not None:
        subtb_losses = jnp.where(
            jnp.abs(subtb_discrepancy) <= huber_delta,
            jnp.square(subtb_discrepancy),
            huber_delta * jnp.abs(subtb_discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        subtb_losses = jnp.square(subtb_discrepancy)

    return jnp.mean(subtb_losses.mean(-1)), (
        trajectories,
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jnp.zeros_like(subtb_losses),
        jax.lax.stop_gradient(subtb_losses),
    )


def loss_fn_joint(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    loss_type: Literal["subtb", "tb_subtb", "lv_subtb"] = "subtb",
    n_chunks: int = 1,
    subtb_weight: float = 1.0,
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
):
    (
        _,
        running_costs,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        log_fs,
        init_fwd_log_probs,
    ) = rnd_partial(key, model_state, params)

    bs, T = log_pfs_over_pbs.shape
    assert T % n_chunks == 0
    chunk_size = T // n_chunks

    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    logZ = params["params"]["logZ"]

    tb_losses = jnp.zeros_like(log_rewards)
    if loss_type != "subtb":
        log_ratio = (
            running_costs - log_rewards * invtemp
        )  # running_costs already contains initial pfs
        if loss_type == "tb_subtb":
            tb_losses = log_ratio + logZ
            logZ = jax.lax.stop_gradient(logZ)
        else:  # loss_type == "lv_subtb"
            tb_losses = log_ratio - jnp.mean(log_ratio)

        if huber_delta is not None:
            tb_losses = jnp.where(
                jnp.abs(tb_losses) <= huber_delta,
                jnp.square(tb_losses),
                huber_delta * jnp.abs(tb_losses) - 0.5 * huber_delta**2,
            )
        else:
            tb_losses = jnp.square(tb_losses)

    log_fs = log_fs.at[:, 0].set(logZ + init_fwd_log_probs)
    log_fs = log_fs.at[:, -1].set(log_rewards * invtemp)

    log_pfs_over_pbs = jax.lax.stop_gradient(log_pfs_over_pbs)

    # db_discrepancy = log_fs[:, :-1] + log_pfs_over_pbs - log_fs[:, 1:]
    # subtb_discrepancy1 = db_discrepancy.reshape(bs, n_chunks, -1).sum(-1)
    # The below is equivalent to the above two lines but avoids numerical instability.
    subtb_discrepancy1 = (
        log_fs[:, :-1:chunk_size]
        + log_pfs_over_pbs.reshape(bs, n_chunks, -1).sum(-1)
        - log_fs[:, chunk_size::chunk_size]
    )

    log_pfs_over_pbs_cumsum = jnp.cumsum(log_pfs_over_pbs[:, ::-1], axis=-1)[:, ::-1]
    subtb_discrepancy2 = (
        log_fs[:, :-1:chunk_size] + log_pfs_over_pbs_cumsum[:, ::chunk_size] - log_fs[:, [-1]]
    ) / jnp.arange(1, n_chunks + 1)[None, ::-1]

    # subtb_discrepancy = subtb_discrepancy1
    # subtb_discrepancy = subtb_discrepancy2
    subtb_discrepancy = jnp.concatenate([subtb_discrepancy1, subtb_discrepancy2], axis=1)

    if huber_delta is not None:
        subtb_losses = jnp.where(
            jnp.abs(subtb_discrepancy) <= huber_delta,
            jnp.square(subtb_discrepancy),
            huber_delta * jnp.abs(subtb_discrepancy) - 0.5 * huber_delta**2,
        )
    else:
        subtb_losses = jnp.square(subtb_discrepancy)

    return jnp.mean(tb_losses + subtb_weight * subtb_losses.mean(-1)), (
        trajectories,
        -log_pfs_over_pbs,  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(tb_losses),
        jax.lax.stop_gradient(subtb_losses),
    )


def loss_fn_subtb_lambda(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    lambda_coef_mat: Array,
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
    flow_only: bool = False,  # if True, only optimize flow
):
    (
        _,
        _,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        log_fs,
        init_fwd_log_probs,
    ) = rnd_partial(key, model_state, params)

    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )

    logZ = params["params"]["logZ"]
    log_fs = log_fs.at[:, 0].set(logZ + init_fwd_log_probs)
    log_fs = log_fs.at[:, -1].set(log_rewards * invtemp)

    diff_logp = log_pfs_over_pbs  # (bs, T)
    diff_logp_padded = jnp.concatenate(
        (jnp.zeros((diff_logp.shape[0], 1)), diff_logp.cumsum(axis=-1)),
        axis=1,
    )  # (bs, T+1)
    A1 = diff_logp_padded[:, None, :] - diff_logp_padded[:, :, None]  # (bs, T+1, T+1)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1  # (bs, T+1, T+1)
    A2 = jnp.triu(A2, k=1) ** 2
    subtb_losses = (A2 * lambda_coef_mat[None, :, :]).sum((1, 2))

    return jnp.mean(subtb_losses), (
        trajectories,
        jax.lax.stop_gradient(-log_pfs_over_pbs),  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jnp.zeros_like(subtb_losses),
        jax.lax.stop_gradient(subtb_losses),
    )


def loss_fn_joint_lambda(
    key: RandomKey,
    model_state: TrainState,
    params: ModelParams,
    rnd_partial: Callable[[RandomKey, TrainState, ModelParams], tuple[Array, ...]],
    lambda_coef_mat: Array,
    loss_type: Literal["subtb", "tb_subtb", "lv_subtb"] = "subtb",
    subtb_weight: float = 1.0,
    invtemp: float = 1.0,
    logr_clip: float = -1e5,
    huber_delta: float | None = None,
):
    (
        _,
        running_costs,
        _,
        terminal_costs,
        trajectories,
        log_pfs_over_pbs,
        log_fs,
        init_fwd_log_probs,
    ) = rnd_partial(key, model_state, params)

    bs, T = log_pfs_over_pbs.shape

    log_rewards = jnp.where(
        -terminal_costs > logr_clip,
        -terminal_costs,
        logr_clip - jnp.log(logr_clip + terminal_costs),
    )
    logZ = params["params"]["logZ"]

    tb_losses = jnp.zeros_like(log_rewards)
    if loss_type != "subtb":
        log_ratio = (
            running_costs - log_rewards * invtemp
        )  # running_costs already contains initial pfs
        if loss_type == "tb_subtb":
            tb_losses = log_ratio + logZ
            logZ = jax.lax.stop_gradient(logZ)
        else:  # loss_type == "lv_subtb"
            tb_losses = log_ratio - jnp.mean(log_ratio)

        if huber_delta is not None:
            tb_losses = jnp.where(
                jnp.abs(tb_losses) <= huber_delta,
                jnp.square(tb_losses),
                huber_delta * jnp.abs(tb_losses) - 0.5 * huber_delta**2,
            )
        else:
            tb_losses = jnp.square(tb_losses)

    log_fs = log_fs.at[:, 0].set(logZ + init_fwd_log_probs)
    log_fs = log_fs.at[:, -1].set(log_rewards * invtemp)

    log_pfs_over_pbs = jax.lax.stop_gradient(log_pfs_over_pbs)

    diff_logp = log_pfs_over_pbs  # (bs, T)
    diff_logp_padded = jnp.concatenate(
        (jnp.zeros((diff_logp.shape[0], 1)), diff_logp.cumsum(axis=-1)),
        axis=1,
    )  # (bs, T+1)
    A1 = diff_logp_padded[:, None, :] - diff_logp_padded[:, :, None]  # (bs, T+1, T+1)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1  # (bs, T+1, T+1)
    A2 = jnp.triu(A2, k=1) ** 2
    subtb_losses = (A2 * lambda_coef_mat[None, :, :]).sum((1, 2))

    return jnp.mean(tb_losses + subtb_weight * subtb_losses), (
        trajectories,
        -log_pfs_over_pbs,  # log(pb(s'->s)/pf(s->s'))
        -terminal_costs,  # log_rewards
        jax.lax.stop_gradient(tb_losses),
        jax.lax.stop_gradient(subtb_losses),
    )
