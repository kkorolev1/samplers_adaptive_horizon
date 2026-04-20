"""Code builds on https://github.com/lollcat/fab-jax"""

import itertools
from functools import partial
from typing import Optional, Tuple

import chex
import jax
import jax.nn as nn
import jax.numpy as jnp
import distrax
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import wandb
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


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


def clf_sigmoid_grid(
    model_state,
    target,
    device,
    is_forward: bool,
    level=None,
    n_grid: int = 80,
    marginal_dims: Tuple[int, int] = (0, 1),
):
    """2D grid of sigmoid(classifier logits); returns numpy X, Y, Z for plotting.

    The network expects states of shape ``(dim,)``; we embed the 2D mesh into the
    full space by fixing non-marginal coordinates to zero (same as density grid).
    """
    dim = int(target.dim)
    d0, d1 = int(marginal_dims[0]), int(marginal_dims[1])
    bounds = (-target._plot_bound, target._plot_bound)
    x = jnp.linspace(*bounds, n_grid)
    y = jnp.linspace(*bounds, n_grid)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    xy = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    grid = jnp.zeros((xy.shape[0], dim), dtype=xy.dtype)
    grid = grid.at[:, d0].set(xy[:, 0]).at[:, d1].set(xy[:, 1])
    grid = jax.device_put(grid, device)

    model = partial(model_state.apply_fn)
    if level is not None:
        l = level * jnp.ones((*grid.shape[:-1], 1))
        model = partial(model, l=l)

    if is_forward:
        clf_logits, *_ = model(model_state.params, grid, predict_fwd=True)
    else:
        clf_logits, *_ = model(model_state.params, grid, predict_fwd=False)
    pdf = nn.sigmoid(clf_logits).reshape(X.shape)
    return np.asarray(X), np.asarray(Y), np.asarray(pdf)


def _marginal_embedded_grid(
    target,
    device,
    marginal_dims: Tuple[int, int],
    n_grid: int,
):
    """Meshgrid in ``(d0, d1)`` embedded in ``R^dim`` (other coords zero)."""
    dim = int(target.dim)
    d0, d1 = int(marginal_dims[0]), int(marginal_dims[1])
    bounds = (-float(target._plot_bound), float(target._plot_bound))
    x = jnp.linspace(bounds[0], bounds[1], n_grid)
    y = jnp.linspace(bounds[0], bounds[1], n_grid)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    xy = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    grid = jnp.zeros((xy.shape[0], dim), dtype=xy.dtype)
    grid = grid.at[:, d0].set(xy[:, 0]).at[:, d1].set(xy[:, 1])
    grid = jax.device_put(grid, device)
    return X, Y, grid


def drift_vector_field_figures(
    model_state,
    target,
    gamma: float,
    device,
    marginal_dims: Tuple[int, int] = (0, 1),
    n_grid: int = 22,
    quiver_stride: int = 2,
    show: bool = False,
):
    """Log forward/backward drift fields ``(mean - s) / gamma`` on a 2D marginal slice.

    Returns wandb image dict keys ``figures/drift_fwd_quiver`` and
    ``figures/drift_bwd_quiver``.
    """
    d0, d1 = int(marginal_dims[0]), int(marginal_dims[1])
    X, Y, grid = _marginal_embedded_grid(
        target, device, marginal_dims, n_grid=n_grid
    )
    model = partial(model_state.apply_fn)
    g = float(gamma)
    if g == 0:
        g = 1e-8

    def _one_figure(predict_fwd: bool, prefix: str):
        if predict_fwd:
            _, mean, *_ = model(model_state.params, grid, predict_fwd=True)
        else:
            _, mean, _ = model(model_state.params, grid, predict_fwd=False)
        drift = (mean - grid) / g
        U = drift[:, d0].reshape(X.shape)
        V = drift[:, d1].reshape(X.shape)
        Xn = np.asarray(X)
        Yn = np.asarray(Y)
        Un = np.asarray(U)
        Vn = np.asarray(V)
        sl = slice(None, None, max(1, int(quiver_stride)))
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        mag = np.hypot(Un, Vn)
        cf = ax.contourf(
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
        ax.set_xlabel(f"x{d0 + 1}")
        ax.set_ylabel(f"x{d1 + 1}")
        b = float(target._plot_bound)
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.35)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="|drift|")
        wb = {f"figures/{prefix}_quiver": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close(fig)
        return wb

    out = {}
    out.update(_one_figure(True, "drift_fwd"))
    out.update(_one_figure(False, "drift_bwd"))
    return out


def _marginal_density_grid(
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


_TRAJ_BG_CMAP = LinearSegmentedColormap.from_list(
    "traj_blue_green", ["#1f77ff", "#2ca02c"]
)


def _plot_gradient_trajectory(ax: plt.Axes, xy: np.ndarray, linewidth: float = 3.5):
    """Blue (start) → green (end) path with white stroke so it reads on heatmaps."""
    if xy.shape[0] < 2:
        ax.scatter(
            xy[0, 0],
            xy[0, 1],
            c=[_TRAJ_BG_CMAP(0.5)],
            s=85,
            edgecolors="black",
            zorder=5,
        )
        return
    pts = xy.reshape(-1, 1, 2)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
    nseg = len(segments)
    lc = LineCollection(
        segments,
        cmap=_TRAJ_BG_CMAP,
        norm=Normalize(0.0, 1.0),
        linewidths=linewidth,
        zorder=4,
        capstyle="round",
    )
    lc.set_array(np.linspace(0.0, 1.0, nseg))
    lc.set_path_effects(
        [patheffects.withStroke(linewidth=linewidth + 2.5, foreground="white")]
    )
    ax.add_collection(lc)
    ax.scatter(
        xy[0, 0],
        xy[0, 1],
        c=[_TRAJ_BG_CMAP(0.0)],
        s=85,
        edgecolors="black",
        zorder=5,
    )
    ax.scatter(
        xy[-1, 0],
        xy[-1, 1],
        c=[_TRAJ_BG_CMAP(1.0)],
        s=85,
        edgecolors="black",
        marker="X",
        zorder=5,
    )


def visualize_trajectory_examples(
    trajectories,
    trajectories_length,
    target,
    model_state,
    dims=(0, 1),
    device="cpu",
    prefix="trajectories_fwd",
    num_examples: int = 6,
    clf_is_forward: bool = True,
    use_classifier_bg: bool = False,
    show: bool = False,
):
    """One figure per example: density background and optional classifier background.

    Keys: ``figures/{prefix}_vis_exNN_density`` and ``figures/{prefix}_vis_exNN_clf``.
    """
    batch_size = int(trajectories.shape[0])
    n = min(int(num_examples), batch_size)
    dim = int(target.dim)
    bounds = (-float(target._plot_bound), float(target._plot_bound))
    d0, d1 = int(dims[0]), int(dims[1])

    x1d, x2d, zd = _marginal_density_grid(target, dim, (d0, d1), bounds, n_points=110)

    Xc = Yc = Zc = None
    if use_classifier_bg and model_state is not None:
        Xc, Yc, Zc = clf_sigmoid_grid(
            model_state,
            target,
            device,
            clf_is_forward,
            n_grid=80,
            marginal_dims=(d0, d1),
        )

    out = {}
    for i in range(n):
        tl = int(trajectories_length[i])
        valid = trajectories[i, :tl]
        xy = np.asarray(jnp.stack([valid[:, d0], valid[:, d1]], axis=1))

        fig_d, ax_d = plt.subplots(figsize=(6, 6))
        ax_d.contourf(x1d, x2d, zd, levels=24, cmap="copper", alpha=0.45)
        _plot_gradient_trajectory(ax_d, xy)
        ax_d.set_xlabel(f"x{d0 + 1}")
        ax_d.set_ylabel(f"x{d1 + 1}")
        ax_d.set_xlim(bounds[0], bounds[1])
        ax_d.set_ylim(bounds[0], bounds[1])
        ax_d.set_aspect("equal", adjustable="box")
        ax_d.grid(True, linestyle="--", alpha=0.35)
        out[f"figures/{prefix}_vis_ex{i:02d}_density"] = [wandb.Image(fig_d)]
        if show:
            plt.show()
        else:
            plt.close(fig_d)

        if use_classifier_bg and model_state is not None and Zc is not None:
            fig_c, ax_c = plt.subplots(figsize=(6, 6))
            ax_c.pcolormesh(Xc, Yc, Zc, cmap="viridis", alpha=0.52, shading="auto")
            _plot_gradient_trajectory(ax_c, xy)
            ax_c.set_xlabel(f"x{d0 + 1}")
            ax_c.set_ylabel(f"x{d1 + 1}")
            ax_c.set_xlim(bounds[0], bounds[1])
            ax_c.set_ylim(bounds[0], bounds[1])
            ax_c.set_aspect("equal", adjustable="box")
            ax_c.grid(True, linestyle="--", alpha=0.35)
            out[f"figures/{prefix}_vis_ex{i:02d}_clf"] = [wandb.Image(fig_c)]
            if show:
                plt.show()
            else:
                plt.close(fig_c)

    return out


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
