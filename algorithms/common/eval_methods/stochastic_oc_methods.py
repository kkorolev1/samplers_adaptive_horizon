from functools import partial

import jax
import jax.numpy as jnp
import jax.nn as nn

from eval import discrepancies
from eval.utils import (
    avg_stddiv_across_marginals,
    compute_reverse_ess,
    moving_averages,
    save_samples,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb


def visualize_clf_heatmap(
    model_state,
    target,
    is_forward=True,
    device="cpu",
    alpha=0.9,
    shrink=1.0,
    prefix="",
    show=False,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    bounds = (-target._plot_bound, target._plot_bound)
    x = jnp.linspace(*bounds, 50)
    y = jnp.linspace(*bounds, 50)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    grid = jax.device_put(grid, device)

    if is_forward:
        clf_logits, *_ = model_state.apply_fn(
            model_state.params,
            grid,
            predict_fwd=True,
        )
    else:
        clf_logits, *_ = model_state.apply_fn(
            model_state.params,
            grid,
            predict_bwd=True,
        )
    pdf = nn.sigmoid(clf_logits).reshape(X.shape)

    im = ax.pcolormesh(X, Y, pdf, cmap="viridis", alpha=alpha, shading="auto")
    cbar = fig.colorbar(
        im, shrink=shrink, label="Classifier probability", fraction=0.046, pad=0.04
    )
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    ax.set_aspect("equal", adjustable="box")

    wb = {f"figures/{prefix + '_' if prefix else ''}vis": [wandb.Image(fig)]}
    if show:
        plt.show()
    else:
        plt.close()
    return wb


def visualize_trajectories(
    trajectories,
    trajectories_length,
    target,
    dims=(0, 1),
    device="cpu",
    alpha=0.8,
    prefix="",
    show=False,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    samples = trajectories[jnp.arange(trajectories.shape[0]), trajectories_length - 1]
    samples = samples[:, dims]
    min_bounds = samples.min(axis=0) - 1.5
    max_bounds = samples.max(axis=0) + 1.5

    batch_size = trajectories.shape[0]

    cmap = plt.get_cmap("viridis")
    colors = cmap(jnp.linspace(0, 1, batch_size))

    marker_size = 50

    x = jnp.linspace(min_bounds[0], max_bounds[0], 50)
    y = jnp.linspace(min_bounds[1], max_bounds[1], 50)

    # if trajectories.shape[-1] == 2:
    #     X, Y = jnp.meshgrid(x, y, indexing="xy")
    #     grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    #     grid = jax.device_put(grid, device)
    #     pdf = jnp.exp(target.log_prob(grid)).reshape(X.shape)
    #     pdf = jax.device_put(pdf, jax.devices("cpu")[0])
    #     levels = jnp.linspace(pdf.min(), pdf.max(), 20)
    #     ax.contourf(X, Y, pdf, levels=levels, cmap="viridis", alpha=0.5)

    # 3. Loop through and plot each trajectory
    for i in range(batch_size):
        color = colors[i]

        # 4. Slice the valid part of the trajectory
        valid_traj = trajectories[i, : trajectories_length[i]]

        # 5. Plot the trajectory line
        ax.plot(
            valid_traj[:, dims[0]],
            valid_traj[:, dims[1]],
            color=color,
            alpha=alpha,
        )

        # 6. Mark the start and end points
        ax.scatter(
            valid_traj[0, dims[0]],
            valid_traj[0, dims[1]],
            color=color,
            marker="o",
            s=marker_size,
            edgecolors="black",
            zorder=3,
        )
        ax.scatter(
            valid_traj[-1, dims[0]],
            valid_traj[-1, dims[1]],
            color=color,
            marker="X",
            s=marker_size,
            edgecolors="black",
            zorder=3,
        )

    ax.set_xlabel(f"x{dims[0]+1}")
    ax.set_ylabel(f"x{dims[1]+1}")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    ax.set_aspect("equal", adjustable="box")

    wb = {f"figures/{prefix + '_' if prefix else ''}vis": [wandb.Image(fig)]}
    if show:
        plt.show()
    else:
        plt.close()
    return wb


def get_eval_fn(rnd, target, target_xs, cfg):
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
        "ESS/forward": [],
        "ESS/reverse": [],
        "discrepancies/mmd": [],
        "discrepancies/sd": [],
        "other/target_log_prob": [],
        "other/delta_mean_marginal_std": [],
        "other/EMC": [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
        "log_var/log_var": [],
        "log_var/traj_bal_ln_z": [],
        "mean_traj_length/reverse": [],
        "mean_traj_length/forward": [],
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
            _,
            trajectories_length,
        ) = rnd_reverse(
            key, model_state, *params
        )[:5]
        samples = trajectories[
            jnp.arange(trajectories.shape[0]), trajectories_length - 1
        ]
        log_is_weights = -running_costs
        ln_z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(cfg.eval_samples)
        elbo = jnp.mean(log_is_weights)
        log_var = jnp.var(running_costs, ddof=0)

        if target.log_Z is not None:
            logger["logZ/delta_reverse"].append(jnp.abs(ln_z - target.log_Z))

        logger["mean_traj_length/reverse"].append(jnp.mean(trajectories_length))
        logger["logZ/reverse"].append(ln_z)
        logger["KL/elbo"].append(elbo)
        logger["ESS/reverse"].append(
            compute_reverse_ess(log_is_weights, cfg.eval_samples)
        )
        logger["other/target_log_prob"].append(jnp.mean(target.log_prob(samples)))
        logger["other/delta_mean_marginal_std"].append(
            jnp.abs(avg_stddiv_across_marginals(samples) - target.marginal_std)
        )
        logger["log_var/log_var"].append(log_var)

        if cfg.compute_forward_metrics and target.can_sample:
            (
                _,
                fwd_running_costs,
                _,
                _,
                fwd_trajectories_length,
            ) = rnd_forward(
                jax.random.PRNGKey(0), model_state, *params
            )[:5]
            fwd_log_is_weights = -fwd_running_costs
            fwd_ln_z = jax.scipy.special.logsumexp(fwd_log_is_weights) - jnp.log(
                cfg.eval_samples
            )
            eubo = jnp.mean(fwd_log_is_weights)

            fwd_ess = jnp.exp(
                fwd_ln_z
                - (
                    jax.scipy.special.logsumexp(fwd_log_is_weights)
                    - jnp.log(cfg.eval_samples)
                )
            )

            if target.log_Z is not None:
                logger["logZ/delta_forward"].append(jnp.abs(fwd_ln_z - target.log_Z))
            logger["logZ/forward"].append(fwd_ln_z)
            logger["KL/eubo"].append(eubo)
            logger["ESS/forward"].append(fwd_ess)
            logger["mean_traj_length/forward"].append(jnp.mean(fwd_trajectories_length))

        logger.update(target.visualise(samples=samples))
        if cfg.target.dim == 2:
            logger.update(
                visualize_clf_heatmap(
                    model_state,
                    target,
                    is_forward=True,
                    device=samples.device,
                    prefix="fwd_clf",
                )
            )
            logger.update(
                visualize_clf_heatmap(
                    model_state,
                    target,
                    is_forward=False,
                    device=samples.device,
                    prefix="bwd_clf",
                )
            )
        logger.update(
            visualize_trajectories(
                trajectories[:10],
                trajectories_length[:10],
                target,
                dims=(0, 1),
                device=samples.device,
                prefix="trajectories",
            )
        )

        if cfg.compute_emc and cfg.target.has_entropy:
            logger["other/EMC"].append(target.entropy(samples))

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
                    logger[f"model_selection/{key}_MAX"].append(max(logger[key]))
                    logger[f"model_selection/{key}_MIN"].append(min(logger[key]))
                else:
                    logger[key] = [value]
                    logger[f"model_selection/{key}_MAX"] = [max(logger[key])]
                    logger[f"model_selection/{key}_MIN"] = [min(logger[key])]

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return short_eval, logger
