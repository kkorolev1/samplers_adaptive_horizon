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
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import wandb
from functools import partial
import matplotlib.patheffects as patheffects
from matplotlib.colors import to_rgba

white_blue_cmap = LinearSegmentedColormap.from_list(
    "white_blue",
    ["#ffffff", "#dbe9ff", "#8fb6ff", "#3b6fb6", "#0b1f4d"],
)


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
    ax.contourf(
        x1,
        x2,
        z,
        levels=levels,
    )


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
    color="#ad102d",
    alpha=0.5,
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
    x = np.linspace(*bounds, 50)
    y = np.linspace(*bounds, 50)
    X, Y = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

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
    pdf = np.array(nn.sigmoid(clf_logits).reshape(X.shape))

    # im = ax.pcolormesh(X, Y, pdf, cmap="viridis", alpha=alpha, shading="auto")
    im = ax.contourf(
        X,
        Y,
        pdf,
        levels=20,
        cmap=white_blue_cmap,
        alpha=1.0,
    )
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
            color=color,
            alpha=alpha,
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


def plot_gradient_trajectory(
    ax: plt.Axes,
    xy: np.ndarray,
    color="red",
    start_marker_color="#e86b95",
    alpha_start: float = 0.4,
    alpha_end: float = 1.0,
    marker_size: float = 40,
    linewidth: float = 1.5,
):
    if xy.shape[0] < 2:
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            c=start_marker_color,
            s=marker_size,
            edgecolors="black",
            zorder=5,
        )
        return
    pts = xy.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    nseg = len(segments)

    base_rgba = np.array(to_rgba(color))
    alphas = np.linspace(alpha_start, alpha_end, nseg)
    colors = np.repeat(base_rgba[None, :], nseg, axis=0)
    colors[:, -1] = alphas

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=linewidth,
        zorder=4,
        capstyle="round",
    )
    stroke_color = base_rgba.copy()
    stroke_color[:3] = np.maximum(stroke_color[:3] - 0.5, 0.0)
    lc.set_path_effects(
        [patheffects.withStroke(linewidth=1.5, foreground=stroke_color)]
    )
    ax.add_collection(lc)
    ax.scatter(
        xy[0, 0],
        xy[0, 1],
        c=start_marker_color,
        s=marker_size,
        edgecolors="black",
        zorder=5,
    )
    # end_color = base_rgba.copy()
    # end_color[:3] = np.maximum(end_color[:3] - 0.3, 0.0)
    ax.scatter(
        xy[-1, 0],
        xy[-1, 1],
        c=start_marker_color,
        s=marker_size,
        edgecolors="black",
        marker="X",
        zorder=5,
    )


def marginal_density_grid(
    target,
    dim: int,
    marginal_dims: Tuple[int, int],
    bounds: tuple[float, float],
    n_points: int = 110,
):
    x_points_dim1 = np.linspace(bounds[0], bounds[1], n_points)
    x_points_dim2 = np.linspace(bounds[0], bounds[1], n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))

    def sliced_log_prob(x_arr):
        xj = jnp.asarray(x_arr)
        _x = jnp.zeros((xj.shape[0], dim))
        _x = _x.at[:, marginal_dims].set(xj)
        return target.log_prob(_x)

    log_probs = sliced_log_prob(x_points)
    log_probs = jnp.clip(log_probs, a_min=-1000, a_max=None)
    z = jnp.exp(log_probs).reshape(n_points, n_points)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    return np.asarray(x1), np.asarray(x2), np.asarray(z)


def visualize_trajectories(
    trajectories,
    trajectories_length,
    target,
    dims=(0, 1),
    color="#ad102d",
    num_examples: int = 3,
    prefix: str = "",
    show: bool = False,
):
    batch_size = int(trajectories.shape[0])
    dim = trajectories[0].shape[-1]
    n = min(int(num_examples), batch_size)
    dim = int(target.dim)
    bounds = (-float(target._plot_bound), float(target._plot_bound))
    d0, d1 = int(dims[0]), int(dims[1])

    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(n):
        tl = int(trajectories_length[i])
        valid = trajectories[i, :tl]
        xy = np.asarray(jnp.stack([valid[:, d0], valid[:, d1]], axis=1))
        plot_gradient_trajectory(ax, xy, color=color)

    x1d, x2d, zd = marginal_density_grid(target, dim, (d0, d1), bounds, n_points=110)
    ax.contourf(
        x1d,
        x2d,
        zd,
        levels=20,
        cmap=white_blue_cmap,
        alpha=1.0,
    )
    ax.set_xlabel(f"x{d0 + 1}")
    ax.set_ylabel(f"x{d1 + 1}")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)

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
    x = np.linspace(*bounds, 50)
    y = np.linspace(*bounds, 50)
    X, Y = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

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
    prod = np.array(jnp.exp(log_prod).reshape(X.shape))

    # im = ax.pcolormesh(X, Y, prod, cmap="viridis", alpha=alpha, shading="auto")
    im = ax.contourf(
        X,
        Y,
        prod,
        levels=20,
        cmap=white_blue_cmap,
        alpha=1.0,
    )
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
    x = np.linspace(*bounds, 50)
    y = np.linspace(*bounds, 50)
    X, Y = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

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
    flow = np.array(jnp.exp(log_flow).reshape(X.shape))

    # im = ax.pcolormesh(X, Y, flow, cmap="viridis", alpha=alpha, shading="auto")
    im = ax.contourf(
        X,
        Y,
        flow,
        levels=20,
        cmap=white_blue_cmap,
        alpha=1.0,
    )
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


def visualize_kernel_drift(
    model_state,
    target,
    cfg,
    is_forward=True,
    dims=(0, 1),
    shrink=1.0,
    quiver_stride: int = 2,
    prefix="",
    show=False,
    fig=None,
    ax=None,
):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()

    bounds = (-target._plot_bound, target._plot_bound)
    x = np.linspace(*bounds, 50)
    y = np.linspace(*bounds, 50)
    X, Y = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    model = partial(model_state.apply_fn)

    _, model_mean, *_ = model(model_state.params, grid, predict_fwd=is_forward)
    gamma = float(cfg.algorithm.model.gamma)
    drift = (model_mean - grid) / gamma
    U = drift[:, dims[0]].reshape(X.shape)
    V = drift[:, dims[1]].reshape(X.shape)
    Xn = np.asarray(X)
    Yn = np.asarray(Y)
    Un = np.asarray(U)
    Vn = np.asarray(V)
    sl = slice(None, None, max(1, int(quiver_stride)))
    mag = np.hypot(Un, Vn)
    im = ax.contourf(
        Xn,
        Yn,
        mag,
        levels=20,
        cmap="magma",
        alpha=0.45,
    )
    ax.quiver(
        Xn[sl, sl],
        Yn[sl, sl],
        Un[sl, sl],
        Vn[sl, sl],
        color="0.2",
        angles="xy",
        scale_units="xy",
        width=0.005,
    )
    cbar = fig.colorbar(
        im, shrink=shrink, label="Kernel drift norm", fraction=0.046, pad=0.04
    )
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

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


def visualize_kernel_std(
    model_state,
    target,
    cfg,
    is_forward=True,
    dims=(0, 1),
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
    x = np.linspace(*bounds, 50)
    y = np.linspace(*bounds, 50)
    X, Y = np.meshgrid(x, y, indexing="xy")
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    model = partial(model_state.apply_fn)
    _, _, model_scale, *_ = model(model_state.params, grid, predict_fwd=is_forward)
    model_scale = np.linalg.norm(model_scale, axis=-1)
    Z = np.array(model_scale.reshape(X.shape))
    im = ax.contourf(
        X,
        Y,
        Z,
        levels=20,
        cmap=white_blue_cmap,
        alpha=1.0,
    )

    cbar = fig.colorbar(
        im, shrink=shrink, label="Kernel std norm", fraction=0.046, pad=0.04
    )
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

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
