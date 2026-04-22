from functools import partial

import jax
import jax.numpy as jnp
import jax.nn as nn

from eval import discrepancies
from eval.utils import (
    avg_stddiv_across_marginals,
    moving_averages,
    save_samples,
)

from utils.plot_utils import visualize_trajectories


def get_eval_fn(rnd, target, target_xs, cfg, visualize_heatmaps_fn=None):
    rnd_reverse = jax.jit(partial(rnd, prior_to_target=True))

    if cfg.compute_forward_metrics and target.can_sample:
        rnd_forward = jax.jit(
            partial(rnd, prior_to_target=False, terminal_xs=target_xs)
        )

    logger = {
        "KL/elbo": [],
        "KL/eubo": [],
        "logZ/delta_forward": [],
        "logZ/forward": [],
        "logZ/delta_reverse": [],
        "logZ/reverse": [],
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
        "mean_traj_length/reverse": [],
        "mean_traj_length/forward": [],
        "max_traj_length/reverse": [],
        "max_traj_length/forward": [],
    }

    def short_eval(model_state, key):
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)
        (
            trajectories,
            running_costs,
            _,
            trajectories_length,
        ) = rnd_reverse(key, model_state, *params)
        samples = trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ]
        log_is_weights = -running_costs
        ln_z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(cfg.eval_samples)
        elbo = jnp.mean(log_is_weights)

        if target.log_Z is not None:
            logger["logZ/delta_reverse"].append(jnp.abs(ln_z - target.log_Z))

        logger["mean_traj_length/reverse"].append(jnp.mean(trajectories_length))
        logger["max_traj_length/reverse"].append(jnp.max(trajectories_length))
        logger["logZ/reverse"].append(ln_z)
        logger["KL/elbo"].append(elbo)

        if cfg.compute_forward_metrics and target.can_sample:
            (
                fwd_trajectories,
                fwd_running_costs,
                _,
                fwd_trajectories_length,
            ) = rnd_forward(jax.random.PRNGKey(0), model_state, *params)[:5]
            fwd_log_is_weights = -fwd_running_costs
            fwd_ln_z = jax.scipy.special.logsumexp(fwd_log_is_weights) - jnp.log(
                cfg.eval_samples
            )
            eubo = jnp.mean(fwd_log_is_weights)

            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_ln_z - target.log_Z))
            logger["logZ/forward"].append(fwd_ln_z)
            logger["KL/eubo"].append(eubo)
            logger["mean_traj_length/forward"].append(jnp.mean(fwd_trajectories_length))
            logger["max_traj_length/forward"].append(jnp.max(fwd_trajectories_length))

        logger.update(target.visualise(samples=samples))
        if cfg.target.dim == 2 and visualize_heatmaps_fn is not None:
            visualize_heatmaps_fn(logger, model_state, target, target_xs, cfg)
        logger.update(
            visualize_trajectories(
                trajectories,
                trajectories_length,
                target,
                dims=(0, 1),
                prefix="trajectories_fwd",
            )
        )
        if cfg.compute_forward_metrics and target.can_sample:
            logger.update(
                visualize_trajectories(
                    fwd_trajectories,
                    fwd_trajectories_length,
                    target,
                    dims=(0, 1),
                    prefix="trajectories_bwd",
                )
            )

        for d in cfg.discrepancies:
            logger[f"discrepancies/{d}"].append(
                getattr(discrepancies, f"compute_{d}")(target_xs, samples, cfg)
                if target_xs is not None
                else jnp.inf
            )

        if cfg.moving_average.use_ma:
            for key, value in moving_averages(
                logger, window_size=cfg.moving_average.window_size
            ).items():
                if isinstance(value, list):
                    value = value[0]
                if key in logger.keys():
                    logger[key].append(value)
                else:
                    logger[key] = [value]

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return short_eval, logger
