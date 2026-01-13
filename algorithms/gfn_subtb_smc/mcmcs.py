import jax
import jax.numpy as jnp


def mala(
    key,
    initial_positions,
    log_prob_fn,
    chain_length,
    initial_step_size,
    num_burnin_steps,
    adapt_step_size=False,
    target_acceptance_rate=0.574,
):
    """
    An improved JIT-compatible MALA sampler with step size adaptation and batching.

    Args:
        key: JAX PRNG key.
        initial_positions: An array of starting points for the chains, with shape (num_chains, dim).
        log_prob_fn: A function that computes the (unnormalized) log probability. This can also be
            the intermediate marginals.
        chain_length: The number of samples to generate per chain after burn-in.
        initial_step_size: The initial step size `h`.
        num_burnin_steps: The number of steps to run the burn-in phase.
        adapt_step_size: Whether to adapt the step size during burn-in.
        target_acceptance_rate: The target acceptance rate for adaptation (default 0.574).

    Returns:
        A tuple containing the generated samples (num_chains, chain_length, dim),
        the final states (num_chains, dim), and their log_reward (num_chains,).
    """

    # Compute value and gradient function
    def grad_log_prob_fn(x):
        log_prob, grad_log_prob = jax.value_and_grad(log_prob_fn)(jax.lax.stop_gradient(x))
        log_prob = jax.lax.stop_gradient(log_prob)
        grad_log_prob = jax.lax.stop_gradient(grad_log_prob)
        return log_prob, grad_log_prob

    num_chains = initial_positions.shape[0]

    def mala_kernel(state, per_step_input):
        """Defines the MALA transition for a single chain."""
        key_gen, position, log_prob, grad_log_prob, step_size, adapt = state
        i = per_step_input

        # 1. Propose a new state using the standard parameterization
        key_proposal, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key_proposal, shape=position.shape)
        proposal_mean = position + step_size * grad_log_prob
        proposal = jax.lax.stop_gradient(proposal_mean + jnp.sqrt(2 * step_size) * noise)
        log_q_forward = -0.25 * jnp.sum((proposal - proposal_mean) ** 2) / step_size

        # 2. Calculate acceptance probability
        proposal_log_prob, proposal_grad_log_prob = grad_log_prob_fn(proposal)
        reverse_proposal_mean = proposal + step_size * proposal_grad_log_prob
        log_q_reverse = -0.25 * jnp.sum((position - reverse_proposal_mean) ** 2) / step_size
        log_alpha = (proposal_log_prob - log_prob) + (log_q_reverse - log_q_forward)

        # 3. Accept or reject
        key_accept, key_gen = jax.random.split(key_gen)
        u = jax.random.uniform(key_accept)
        is_accepted = jnp.log(u) < log_alpha

        new_position = jnp.where(is_accepted, proposal, position)
        new_log_prob = jnp.where(is_accepted, proposal_log_prob, log_prob)
        new_grad_log_prob = jnp.where(is_accepted, proposal_grad_log_prob, grad_log_prob)
        acceptance_prob = jnp.exp(jnp.minimum(0.0, log_alpha))

        # 4. Adapt step size (Robbins-Monro)
        # Adaptation rate diminishes over the burn-in period
        new_step_size = jax.lax.cond(
            adapt,
            lambda _: jnp.maximum(
                1e-6,  # Ensure step size > 0
                step_size + (1.0 / (1 + i)) * (acceptance_prob - target_acceptance_rate),
            ),
            lambda _: step_size,
            operand=None,
        )

        return (
            key_gen,
            new_position,
            new_log_prob,
            new_grad_log_prob,
            new_step_size,
            adapt,
        ), is_accepted

    def mala_single_scan(key, init_position, init_step_size, length, adapt=False):
        init_log_prob, init_grad_log_prob = grad_log_prob_fn(init_position)
        (_, last_position, last_log_prob, _, last_step_size, _), accept_flags = jax.lax.scan(
            mala_kernel,
            (key, init_position, init_log_prob, init_grad_log_prob, init_step_size, adapt),
            jnp.arange(length),
        )
        return last_position, last_log_prob, last_step_size, accept_flags.mean()

    # ---- Execution ----

    burnin_key, sampling_key = jax.random.split(key)

    # 1. Burn-in phase with adaptation
    # We need to pass the loop index `i` to the kernel for adaptation
    # A common trick is to scan over `(state, i)`

    keys = jax.random.split(burnin_key, num_chains)
    initial_step_sizes = jnp.full(num_chains, initial_step_size)

    def _run_burnin():
        return jax.vmap(
            mala_single_scan,
            in_axes=(0, 0, 0, None, None),
        )(keys, initial_positions, initial_step_sizes, num_burnin_steps, adapt_step_size)

    def _skip_burnin():
        return (initial_positions, jnp.zeros(num_chains), initial_step_sizes, jnp.zeros(num_chains))

    burnin_positions, _, burnin_step_sizes, _ = jax.lax.cond(
        num_burnin_steps > 0,
        lambda _: _run_burnin(),
        lambda _: _skip_burnin(),
        operand=None,
    )

    keys = jax.random.split(sampling_key, num_chains)
    final_positions, final_log_probs, _, final_acceptance_rates = jax.vmap(
        mala_single_scan,
        in_axes=(0, 0, 0, None, None),
    )(keys, burnin_positions, burnin_step_sizes, chain_length, False)

    return final_positions, final_log_probs, final_acceptance_rates
