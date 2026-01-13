from functools import partial

import jax
import jax.numpy as jnp

from algorithms.common.types import Array


def per_sample_rnd(
    seed,
    model_state,
    params,
    sde_tuple,
    target,
    num_steps,
    noise_schedule,
    use_lp,
    stop_grad=False,
    prior_to_target=True,
    terminal_x: Array | None = None,
):
    dim, ref_log_prob = sde_tuple
    target_log_prob = target.log_prob

    def langevin_init_fn(x, t, T, target_log_prob):
        tr = t / T
        return (1 - tr) * target_log_prob(x)

    sigmas = noise_schedule
    langevin_init = partial(langevin_init_fn, T=num_steps, target_log_prob=target_log_prob)
    dt = 1.0 / num_steps

    def simulate_prior_to_target(state, per_step_input):
        x, sigma_int, key_gen = state
        step = per_step_input

        step = step.astype(jnp.float32)
        if stop_grad:
            x = jax.lax.stop_gradient(x)

        # Compute SDE components
        sigma_t = sigmas(step)
        sigma_int += sigma_t**2 * dt
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        else:
            langevin = jnp.zeros(x.shape[0])
        model_output, _ = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)
        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x.shape), -4, 4)

        # Euler-Maruyama integration of the SDE
        x_new = x + sigma_t * model_output * dt + sigma_t * noise * jnp.sqrt(dt)

        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = 0.5 * jnp.square(jnp.linalg.norm(model_output)) * dt
        stochastic_cost = (model_output * noise).sum() * jnp.sqrt(dt)

        next_state = (x_new, sigma_int, key_gen)
        per_step_output = (running_cost, stochastic_cost, x_new)
        return next_state, per_step_output

    def simulate_target_to_prior(state, per_step_input):
        """
        Takes samples from the target and moves them to the prior
        """
        x_new, sigma_int, key_gen = state
        step = per_step_input
        t_next = (num_steps - (step - 1)) / num_steps

        sigma_t = sigmas(step)
        sigma_int += sigma_t**2 * dt
        shrink = (t_next - dt) / t_next

        step = step.astype(jnp.float32)
        if stop_grad:
            x_new = jax.lax.stop_gradient(x_new)

        key, key_gen = jax.random.split(key_gen)
        noise = jnp.clip(jax.random.normal(key, shape=x_new.shape), -4, 4)

        x = shrink * x_new + noise * sigma_t * jnp.sqrt(shrink * dt)

        # Compute SDE components
        if use_lp:
            langevin = jax.lax.stop_gradient(jax.grad(langevin_init)(x, step))
        else:
            langevin = jnp.zeros(x.shape[0])
        model_output, _ = model_state.apply_fn(params, x, step * jnp.ones(1), langevin)

        # Compute (running) Radon-Nikodym derivative components
        running_cost = 0.5 * jnp.square(jnp.linalg.norm(model_output)) * dt
        fwd_noise = (1 / (sigma_t * jnp.sqrt(dt))) * (x_new - (x + sigma_t * dt * model_output))
        stochastic_cost = (model_output * fwd_noise).sum() * jnp.sqrt(dt)

        next_state = (x, sigma_int, key_gen)
        per_step_output = (running_cost, stochastic_cost, x)
        return next_state, per_step_output

    key, key_gen = jax.random.split(seed)
    if prior_to_target:
        init_x = jnp.zeros(dim)
        aux = (init_x, jnp.array(0.0), key)
        aux, per_step_output = jax.lax.scan(
            simulate_prior_to_target, aux, jnp.arange(1, num_steps + 1)[::-1]
        )
        terminal_x, final_sigma, _ = aux
    else:
        if terminal_x is None:
            terminal_x = jnp.squeeze(target.sample(key, (1,)))
        key, key_gen = jax.random.split(key_gen)
        aux = (terminal_x, jnp.array(0.0), key)
        aux, per_step_output = jax.lax.scan(
            simulate_target_to_prior, aux, jnp.arange(1, num_steps + 1)
        )
        init_x, final_sigma, _ = aux

    terminal_cost = ref_log_prob(terminal_x, jnp.sqrt(final_sigma)) - target_log_prob(terminal_x)
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
    noise_schedule,
    use_lp,
    stop_grad=False,
    prior_to_target=True,
    terminal_xs: Array | None = None,
):
    seeds = jax.random.split(key, num=batch_size)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t = jax.vmap(
        per_sample_rnd, in_axes=(0, None, None, None, None, None, None, None, None, None, 0)
    )(
        seeds,
        model_state,
        params,
        initial_density_tuple,
        target,
        num_steps,
        noise_schedule,
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
    noise_schedule,
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
        noise_schedule,
        use_lp,
        stop_grad,
    )
    samples, running_costs, _, terminal_costs = aux
    neg_elbo_vals = running_costs + terminal_costs
    return jnp.mean(neg_elbo_vals), (neg_elbo_vals, samples)
