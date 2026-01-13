from functools import partial
from typing import Callable

import chex
import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from algorithms.gfn_subtb.gfn_subtb_rnd import get_flow_bias, ref_log_prob_pinned_brownian


def visualise_true_intermediate_distribution(
    visualize_fn,
    plot_steps: list[int],
    num_steps: int,
    reference_process: str,
    noise_schedule: Callable[[float], float],
    lambda_fn: Callable[[int], float],
    init_std: float,
    target_log_prob_fn: Callable[[chex.Array], chex.Array],
    target_log_prob_t_fn: Callable[[chex.Array, float, float, float], chex.Array],
) -> dict:
    vis_dict = {}

    for step in plot_steps:
        t = step / num_steps

        # Define the mean and covariance factors
        _log_prob_t_fn = None
        if reference_process == "pinned_brownian":
            # TODO
            raise NotImplementedError
        elif reference_process == "ou":
            # TODO
            raise NotImplementedError
        elif reference_process == "ou_dds":
            if t == 1.0:
                _log_prob_t_fn = target_log_prob_fn
            else:
                lambda_t = lambda_fn(step)
                _log_prob_t_fn = partial(target_log_prob_t_fn, lambda_t=lambda_t, init_std=init_std)
        else:
            raise ValueError(f"Reference process {reference_process} not supported.")

        assert _log_prob_t_fn is not None

        vis_dict.update(visualize_fn(samples=None, prefix=f"t={t}", log_prob_fn=_log_prob_t_fn))
    return vis_dict


def visualise_intermediate_distribution(
    visualize_fn,
    plot_steps: list[int],
    num_steps: int,
    trajectories: chex.Array | None,
    model_state: TrainState,
    partial_energy: bool,
    batch_size: int,
    reference_process: str,  # for flow bias
    noise_schedule: Callable[[float], float],  # for flow bias
    beta_fn: Callable[[int], float],  # for flow bias
    initial_dist: distrax.Distribution | None,  # for flow bias
    target_log_prob_fn: Callable[[chex.Array], chex.Array],  # for flow bias
) -> dict:
    vis_dict = {}

    for step in plot_steps:
        t = step / num_steps
        intermediate_states = None
        if trajectories is not None:
            intermediate_states = trajectories[:, step]

        # Define flow_bias function
        flow_bias_fn = None
        if partial_energy:
            if reference_process == "pinned_brownian":
                sigma_t = noise_schedule(step)
                ref_log_prob_fn: Callable[[chex.Array], chex.Array] = partial(
                    ref_log_prob_pinned_brownian, t=t, sigma_t=sigma_t
                )
            elif reference_process in ["ou", "ou_dds"]:
                assert initial_dist is not None
                ref_log_prob_fn: Callable[[chex.Array], chex.Array] = initial_dist.log_prob
            else:
                raise ValueError(f"Reference process {reference_process} not supported.")
            flow_bias_fn: Callable[[chex.Array], chex.Array] = lambda x: get_flow_bias(
                beta_fn(step), ref_log_prob_fn(x), target_log_prob_fn(x)
            )

        def intermediate_log_prob_fn(x: chex.Array) -> chex.Array:
            batched = x.ndim == 2
            if not batched:
                x = x[None]

            time_arr = t * jnp.ones(1)

            def single_log_f(s: chex.Array) -> chex.Array:
                _, log_f_i = model_state.apply_fn(
                    model_state.params,
                    s,
                    time_arr,
                    jnp.zeros(s.shape[0]),  # langevin term doesn't affect the flows
                )
                if flow_bias_fn is not None:
                    log_f_i = log_f_i + flow_bias_fn(s)
                return log_f_i

            batched_log_f = jax.jit(jax.vmap(single_log_f, in_axes=(0,)))

            n = x.shape[0]
            if n <= batch_size:
                log_f = batched_log_f(x)
            else:
                outputs = []
                for start in range(0, n, batch_size):
                    stop = min(start + batch_size, n)
                    outputs.append(batched_log_f(x[start:stop]))
                log_f = jnp.concatenate(outputs, axis=0)

            if not batched:
                log_f = jnp.squeeze(log_f, axis=0)

            return log_f

        vis_dict.update(
            visualize_fn(intermediate_states, prefix=f"t={t}", log_prob_fn=intermediate_log_prob_fn)
        )
    return vis_dict
