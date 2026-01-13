import jax
import jax.numpy as jnp

from algorithms.common.types import Array


def cos_sq_fn_step_scheme(n_steps, s=0.008, noise_scale=6.0, dtype=jnp.float32):
    pre_phase = jnp.linspace(0, 1, n_steps + 1, dtype=dtype)
    phase = ((pre_phase + s) / (1 + s)) * jnp.pi * 0.5
    dts = jnp.cos(phase) ** 4
    dts_out = dts / dts.sum()
    return dts_out * noise_scale


def per_sample_rnd(
    seed,
    model_state,
    params,
    initial_density_tuple,
    target,
    num_steps,
    use_lp,
    stop_grad=False,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    init_std, init_sampler, init_log_prob, noise_scale = initial_density_tuple
    target_log_prob = target.log_prob

    langevin_init = lambda x: target_log_prob(x)
    betas = cos_sq_fn_step_scheme(num_steps, noise_scale=noise_scale)[::-1]

    def simulate_prior_to_target(state, per_step_input):
        x, key_gen = state
        step = per_step_input
        beta_t = jnp.clip(jnp.sqrt(betas[step]), 0, 1)
        alpha_t = jnp.sqrt(1 - beta_t**2)
        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x))
        else:
            langevin = jnp.zeros(x.shape[0])
        model_output, _ = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)
        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x.shape), -4, 4)

        # Exponential integration of the SDE
        x_new = alpha_t * x + beta_t**2 * model_output + beta_t * noise * init_std

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = (
            0.5 * beta_t**2 * jnp.square(jnp.linalg.norm(model_output)) * (1 / init_std**2)
        )
        stochastic_cost = (model_output * noise).sum() * beta_t / init_std

        next_state = (x_new, key_gen)
        per_step_output = (running_cost, stochastic_cost, x_new)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        x_new, key_gen = state
        step = per_step_input
        beta_t = jnp.clip(jnp.sqrt(betas[step]), 0, 1)
        alpha_t = jnp.sqrt(1 - beta_t**2)
        step = step.astype(jnp.float32)
        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x_new.shape), -4, 4)

        # Exponential integration of the SDE
        x = alpha_t * x_new + beta_t * noise * init_std
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x))
        else:
            langevin = jnp.zeros(x.shape[0])
        model_output, _ = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = (
            0.5 * beta_t**2 * jnp.square(jnp.linalg.norm(model_output)) * (1 / init_std**2)
        )
        fwd_noise = (1 / (init_std * beta_t)) * (x_new - (alpha_t * x + beta_t**2 * model_output))
        stochastic_cost = (model_output * fwd_noise).sum() * beta_t / init_std

        next_state = (x, key_gen)
        per_step_output = (running_cost, stochastic_cost, x)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.clip(init_sampler(seed=key), -4 * init_std, 4 * init_std)
        key, key_gen = jax.random.split(key_gen)
        aux = (init_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(1, num_steps + 1)[::-1]
        )
        terminal_x, _ = aux
    else:
        # Initialize from provided terminal state if available; otherwise sample from target
        if terminal_x is None:
            terminal_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (terminal_x, key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(1, num_steps + 1)
        )
        init_x, _ = aux

    terminal_cost = init_log_prob(terminal_x) - target_log_prob(terminal_x)
    running_cost, stochastic_cost, x_t = per_step_output
    return terminal_x, running_cost, stochastic_cost, terminal_cost, x_t


def rnd(
    key,
    model_state,
    params,
    batch_size,
    initial_density_tuple,
    target,
    num_steps,
    use_lp,
    stop_grad=False,
    prior_to_target=True,
    terminal_xs: Array | None = None,
):
    seeds = jax.random.split(key, num=batch_size)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t = jax.vmap(
        per_sample_rnd, in_axes=(0, None, None, None, None, None, None, None, None, 0)
    )(
        seeds,
        model_state,
        params,
        initial_density_tuple,
        target,
        num_steps,
        use_lp,
        stop_grad,
        prior_to_target,
        terminal_xs,
    )

    return x_0, running_costs.sum(1), stochastic_costs.sum(1), terminal_costs


def neg_elbo(
    key,
    model_state,
    params,
    batch_size,
    initial_density,
    target_density,
    num_steps,
    use_lp,
    stop_grad=False,
):
    aux = rnd(
        key,
        model_state,
        params,
        batch_size,
        initial_density,
        target_density,
        num_steps,
        use_lp,
        stop_grad,
    )
    samples, running_costs, _, terminal_costs = aux
    neg_elbo_vals = running_costs + terminal_costs
    return jnp.mean(neg_elbo_vals), (neg_elbo_vals, samples)
