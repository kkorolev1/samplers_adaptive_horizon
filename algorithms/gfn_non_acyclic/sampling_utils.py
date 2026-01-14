import warnings
from functools import partial
from typing import Callable, Literal

import chex
import jax
import jax.numpy as jnp
from jax import random


def ess(
    log_iws: chex.Array | None = None,  # (bs,)
    normalized_weights: chex.Array | None = None,  # (bs,)
) -> chex.Array:
    if normalized_weights is None:
        assert log_iws is not None
        normalized_weights = jax.nn.softmax(log_iws, axis=0)  # (bs,)
    return 1 / (normalized_weights**2).sum()  # scalar


@jax.jit
def binary_search_smoothing(
    log_iws: chex.Array,
    target_ess: float = 0.0,
    tol=1e-3,
    max_steps=1000,
) -> tuple[chex.Array, float]:
    batch_size = log_iws.shape[0]  # type: ignore

    # Helper closures (JAX-friendly)
    def normalize_ess(_log_iws: chex.Array) -> chex.Array:
        return ess(log_iws=_log_iws) / batch_size

    tol_f = jnp.asarray(tol, dtype=log_iws.dtype)

    # Early exit if already meets target ESS
    init_norm_ess = normalize_ess(log_iws)

    def _early_return(_: None):
        return log_iws, jnp.asarray(1.0, dtype=log_iws.dtype)

    def _continue(_: None):
        # Expand search range so that ESS(log_iws / search_max) >= target_ess
        init_search_min = jnp.asarray(1.0, dtype=log_iws.dtype)
        init_search_max = jnp.asarray(10.0, dtype=log_iws.dtype)

        def range_cond_fun(state):
            _, search_max, _ = state
            cur_ess = normalize_ess(log_iws / search_max)
            return cur_ess < target_ess

        def range_body_fun(state):
            search_min, search_max, steps = state
            return search_min * 10.0, search_max * 10.0, steps + 1

        search_min, search_max, _ = jax.lax.while_loop(
            range_cond_fun,
            range_body_fun,
            (init_search_min, init_search_max, jnp.asarray(0, dtype=jnp.int32)),
        )

        # Binary search for temperature achieving ESS close to target
        init_state = (
            search_min,  # search_min
            search_max,  # search_max
            jnp.asarray(0, dtype=jnp.int32),  # steps
            jnp.asarray(False),  # done
            jnp.asarray(1.0, dtype=log_iws.dtype),  # final_temp
        )

        def bin_cond_fun(state):
            _, _, step, done, _ = state
            return (~done) & (step < max_steps)

        def bin_body_fun(state):
            search_min, search_max, step, _, _ = state
            mid = (search_min + search_max) / 2.0

            tempered_log_iws = log_iws / mid
            new_ess = normalize_ess(tempered_log_iws)
            is_converged = jnp.abs(new_ess - target_ess) < tol_f

            # Update the bracket based on whether ESS is above or below target
            go_left = new_ess > target_ess
            new_search_max = jnp.where(go_left, mid, search_max)
            new_search_min = jnp.where(go_left, search_min, mid)

            return new_search_min, new_search_max, step + 1, is_converged, mid

        search_min, search_max, step, done, final_temp = jax.lax.while_loop(
            bin_cond_fun, bin_body_fun, init_state
        )

        # print warning if not converged
        def _print_warning(_: None):
            jax.debug.print(f"Binary search failed in {max_steps} steps")
            return None

        jax.lax.cond(step >= max_steps, _print_warning, lambda _: None, operand=None)

        # Choose final temperature: best match if found, else mid of bracket
        return log_iws / final_temp, final_temp

    return jax.lax.cond(init_norm_ess >= target_ess, _early_return, _continue, operand=None)


def multinomial(
    key: jax.Array, weights: chex.Array, N: int, replacement: bool = True
) -> chex.Array:
    """Return sampled indices from multinomial distribution.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
        replacement: Whether to sample with replacement.
    """
    indices = jnp.arange(len(weights))
    return random.choice(
        key,
        indices,
        shape=(N,),
        replace=replacement,
        p=jax.nn.softmax(weights),
    )


def stratified(key: jax.Array, weights: chex.Array, N: int, replacement: bool = True) -> chex.Array:
    """Return sampled indices using stratified (re)sampling technique.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
    """
    if not replacement:
        warnings.warn(
            "Stratified sampling does not support sampling without replacement. "
            "Using multinomial sampling instead."
        )
        return multinomial(key, weights, N, replacement=True)

    # Normalize weights if they're not already normalized
    weights = weights / weights.sum()

    cumsum = jnp.cumsum(weights)
    u = (jnp.arange(N) + random.uniform(key, shape=(N,))) / N
    indices = jnp.searchsorted(cumsum, u)
    return jnp.clip(indices, 0, len(weights) - 1)


def systematic(key: jax.Array, weights: chex.Array, N: int, replacement: bool = True) -> chex.Array:
    """Return sampled indices using systematic (re)sampling technique.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
    """
    if not replacement:
        warnings.warn(
            "Systematic sampling does not support sampling without replacement. "
            "Using multinomial sampling instead."
        )
        return multinomial(key, weights, N, replacement=True)

    # Normalize weights
    weights = weights / weights.sum()

    cumsum = jnp.cumsum(weights)
    u = (jnp.arange(N) + random.uniform(key, shape=(1,))) / N
    indices = jnp.searchsorted(cumsum, u)
    return jnp.clip(indices, 0, len(weights) - 1)


def rank(
    key: jax.Array,
    weights: chex.Array,
    N: int,
    replacement: bool = True,
    rank_k: float = 0.01,
) -> chex.Array:
    """Return sampled indices using rank-based (re)sampling technique.

    Args:
        key: JAX PRNG key.
        weights: chex.Array of shape (bs,).
        N: Number of samples to draw.
        replacement: Whether to sample with replacement.
        rank_k: A hyperparameter for rank-based sampling.
    """
    ranks = jnp.argsort(jnp.argsort(-weights))  # type: ignore
    new_weights = 1.0 / (rank_k * len(weights) + ranks)
    return multinomial(key, new_weights, N, replacement=replacement)


def get_sampling_func(
    sampling_strategy: Literal["multinomial", "stratified", "systematic", "rank"],
    rank_k: float = 0.01,
) -> Callable[[jax.Array, chex.Array, int, bool], chex.Array]:
    """Factory function to get the desired sampling method.

    Args:
        sampling_strategy: The name of the sampling strategy.
        rank_k: A hyperparameter for rank-based sampling, used only if strategy is 'rank'.

    Returns:
        A callable sampling function.
    """
    if sampling_strategy == "multinomial":
        return multinomial
    elif sampling_strategy == "stratified":
        return stratified
    elif sampling_strategy == "systematic":
        return systematic
    elif sampling_strategy == "rank":
        return partial(rank, rank_k=rank_k)
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
