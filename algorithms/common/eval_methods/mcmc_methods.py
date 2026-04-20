from functools import partial

import jax
import jax.numpy as jnp
import jax.nn as nn
import matplotlib.pyplot as plt
import wandb

from eval import discrepancies
from eval.utils import (
    save_samples,
)

from utils.plot_utils import visualize_trajectories


def get_eval_fn(rnd, target, target_xs, cfg):
    rnd_reverse = jax.jit(partial(rnd, prior_to_target=True))

    logger = {
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
        "mean_traj_length/reverse": [],
        "max_traj_length/reverse": [],
    }

    def short_eval(model_state, key):
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)
        (
            trajectories,
            _,
            _,
            trajectories_length,
        ) = rnd_reverse(key, model_state, *params)
        samples = trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ]
        steps = jnp.linspace(1, trajectories.shape[1] - 1, 15, dtype=int)
        logger["mean_traj_length/reverse"].append(jnp.mean(trajectories_length))
        logger["max_traj_length/reverse"].append(jnp.max(trajectories_length))

        logger.update(target.visualise(samples=samples))
        logger.update(
            visualize_trajectories(
                trajectories[:10],
                trajectories_length[:10],
                target,
                dims=(0, 1),
                device=samples.device,
                prefix="trajectories_fwd",
            )
        )

        for d in cfg.discrepancies:
            values = []
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot()
            for s in steps:
                samples_on_step = trajectories[:, s]
                value = (
                    getattr(discrepancies, f"compute_{d}")(
                        target_xs, samples_on_step, cfg
                    )
                    if target_xs is not None
                    else jnp.inf
                )
                values.append(value)
            values = jnp.array(values)
            ax.plot(steps, values)
            ax.set_title(d)
            ax.set_xlabel("step")
            ax.set_ylabel(d)
            ax.grid()
            logger.update({f"figures/{d}": [wandb.Image(fig)]})
            plt.close(fig)
            logger[f"discrepancies/{d}"].append(values[-1])

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        plt.close("all")

        return logger

    return short_eval, logger
