"""Code builds on https://github.com/lollcat/fab-jax"""

import itertools
from typing import Optional, Tuple

import chex
import jax
import jax.nn as nn
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import wandb
from functools import partial


def plot_contours_2D(
    log_prob_func,
    dim: int,
    ax: plt.Axes,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: tuple[float, float] = (-3, 3),
    levels: int = 20,
    n_points: int = 200,
    log: bool = False,
):
    """Plot the contours of a 2D log prob function."""
    x_points_dim1 = np.linspace(bounds[0], bounds[1], n_points)
    x_points_dim2 = np.linspace(bounds[0], bounds[1], n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))

    def sliced_log_prob(x: chex.Array):
        _x = jnp.zeros((x.shape[0], dim))
        _x = _x.at[:, marginal_dims].set(x)
        return log_prob_func(_x)

    log_probs = sliced_log_prob(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    if log:
        z = log_probs
    else:
        z = jnp.exp(log_probs)
    z = z.reshape(n_points, n_points)
    ax.contourf(x1, x2, z, levels=levels)


def plot_marginal_pair(
    samples: chex.Array,
    ax: plt.Axes,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[float, float] = (-5, 5),
    alpha: float = 0.5,
    max_points: int = 500,
    color: str = "r",
    marker: str = "x",
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.scatter(
        samples[:max_points, marginal_dims[0]],
        samples[:max_points, marginal_dims[1]],
        color=color,
        alpha=alpha,
        marker=marker,
    )


def visualize_clf_heatmap(
    model_state,
    target,
    cfg,
    is_forward=True,
    level=None,
    dims=(0, 1),
    device="cpu",
    alpha=0.9,
    shrink=1.0,
    prefix="",
    show=False,
    fig=None,
    ax=None,
):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()

    bounds = (-target._plot_bound, target._plot_bound)
    x = jnp.linspace(*bounds, 50)
    y = jnp.linspace(*bounds, 50)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    grid = jax.device_put(grid, device)

    model = partial(model_state.apply_fn)

    if level is not None:
        l = level * jnp.ones((*grid.shape[:-1], 1))
        model = partial(model, l=l)

    if is_forward:
        clf_logits, *_ = model(
            model_state.params,
            grid,
            predict_fwd=True,
        )
    else:
        clf_logits, *_ = model(
            model_state.params,
            grid,
            predict_fwd=False,
        )
    pdf = nn.sigmoid(clf_logits).reshape(X.shape)

    im = ax.pcolormesh(X, Y, pdf, cmap="viridis", alpha=alpha, shading="auto")
    cbar = fig.colorbar(
        im, shrink=shrink, label="Classifier probability", fraction=0.046, pad=0.04
    )
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    if level is None:
        if is_forward:
            samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))
        else:
            dim = cfg.target.dim
            initial_dist = distrax.MultivariateNormalDiag(
                jnp.zeros(dim), jnp.ones(dim) * cfg.algorithm.init_std
            )
            samples = initial_dist.sample(
                seed=jax.random.PRNGKey(0), sample_shape=(cfg.eval_samples,)
            )
        plot_marginal_pair(
            samples[:, dims],
            ax,
            marginal_dims=dims,
            bounds=bounds,
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


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


def visualize_trajectories(
    trajectories,
    trajectories_length,
    target,
    dims=(0, 1),
    device="cpu",
    alpha=0.95,
    prefix="",
    show=False,
    fig=None,
    ax=None,
    start_color="#ffe84a",  # bright yellow
    end_color="#d41111",  # deep red
    linewidth=2.5,
    start_marker_size=45,
    end_marker_size=55,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Convert to numpy for matplotlib
    trajectories = np.asarray(trajectories)
    trajectories_length = np.asarray(trajectories_length)

    bounds = (-target._plot_bound, target._plot_bound)
    batch_size = trajectories.shape[0]

    # Optional contour background
    if trajectories.shape[-1] == 2:
        plot_contours_2D(
            target.log_prob,
            trajectories.shape[-1],
            ax,
            marginal_dims=dims,
            bounds=bounds,
            levels=50,
        )

    # A high-contrast warm gradient for dark/cool backgrounds
    traj_cmap = LinearSegmentedColormap.from_list(
        "traj_warm",
        [start_color, "#ff9f1c", end_color],
        N=256,
    )

    for i in range(batch_size):
        traj_len = int(trajectories_length[i])
        if traj_len <= 0:
            continue

        valid_traj = trajectories[i, :traj_len]
        points = valid_traj[:, list(dims)]

        # Draw gradient line only if at least 2 points exist
        if traj_len > 1:
            segments = np.concatenate(
                [points[:-1, None, :], points[1:, None, :]],
                axis=1,
            )

            # One value per segment, smoothly increasing from 0 to 1
            color_values = np.linspace(0.0, 1.0, traj_len - 1)

            lc = LineCollection(
                segments,
                cmap=traj_cmap,
                array=color_values,
                linewidths=linewidth,
                alpha=alpha,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
            ax.add_collection(lc)

        # Start point
        ax.scatter(
            points[0, 0],
            points[0, 1],
            color=start_color,
            marker="o",
            s=start_marker_size,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

        # End point
        ax.scatter(
            points[-1, 0],
            points[-1, 1],
            color=end_color,
            marker="X",
            s=end_marker_size,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

    ax.set_xlabel(f"x{dims[0] + 1}")
    ax.set_ylabel(f"x{dims[1] + 1}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal", adjustable="box")

    wb = {f"figures/{prefix + '_' if prefix else ''}vis": [wandb.Image(fig)]}

    if show:
        plt.show()
    else:
        plt.close(fig)

    return wb


def visualize_flow_clf_heatmap(
    model_state,
    target,
    level=None,
    device="cpu",
    alpha=0.9,
    shrink=1.0,
    prefix="",
    show=False,
    fig=None,
    ax=None,
):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()

    bounds = (-target._plot_bound, target._plot_bound)
    x = jnp.linspace(*bounds, 50)
    y = jnp.linspace(*bounds, 50)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    grid = jax.device_put(grid, device)

    log_reward = target.log_prob(grid)
    model = partial(model_state.apply_fn)

    if level is not None:
        l = level * jnp.ones((*grid.shape[:-1], 1))
        model = partial(model, l=l)

    fwd_clf_logits, *_ = model(
        model_state.params,
        grid,
        log_reward=log_reward,
        predict_fwd=True,
    )
    bwd_clf_logits, *_ = model(
        model_state.params,
        grid,
        predict_fwd=False,
    )
    log_flow = log_reward - nn.log_sigmoid(fwd_clf_logits)
    log_prod = log_flow + nn.log_sigmoid(bwd_clf_logits)
    prod = jnp.exp(log_prod).reshape(X.shape)

    im = ax.pcolormesh(X, Y, prod, cmap="viridis", alpha=alpha, shading="auto")
    cbar = fig.colorbar(
        im,
        shrink=shrink,
        label="F(s) * D_B(s)",
        fraction=0.046,
        pad=0.04,
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


def visualize_flow_heatmap(
    model_state,
    target,
    level=None,
    device="cpu",
    alpha=0.9,
    shrink=1.0,
    prefix="",
    show=False,
    fig=None,
    ax=None,
):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()

    bounds = (-target._plot_bound, target._plot_bound)
    x = jnp.linspace(*bounds, 50)
    y = jnp.linspace(*bounds, 50)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    grid = jax.device_put(grid, device)

    log_reward = target.log_prob(grid)
    model = partial(model_state.apply_fn)

    if level is not None:
        l = level * jnp.ones((*grid.shape[:-1], 1))
        model = partial(model, l=l)

    fwd_clf_logits, *_ = model(
        model_state.params,
        grid,
        predict_fwd=True,
    )
    log_flow = log_reward - nn.log_sigmoid(fwd_clf_logits)
    flow = jnp.exp(log_flow).reshape(X.shape)

    im = ax.pcolormesh(X, Y, flow, cmap="viridis", alpha=alpha, shading="auto")
    cbar = fig.colorbar(
        im,
        shrink=shrink,
        label="F(s)",
        fraction=0.046,
        pad=0.04,
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
