import math
import matplotlib.pyplot as plt
from utils.plot_utils import (
    visualize_clf_heatmap,
    visualize_flow_clf_heatmap,
    visualize_flow_heatmap,
)
import wandb
import numpy as np


def get_invtemp(it: int, n_epochs: int, invtemp: float, invtemp_anneal: bool) -> float:
    if not invtemp_anneal:
        return invtemp
    return linear_annealing(
        it, int(0.5 * n_epochs), invtemp, 1.0, descending=False, avoid_zero=True
    )


def linear_annealing(
    current: int,
    n_rounds: int,
    min_val: float,
    max_val: float,
    exp=True,
    descending=False,
    avoid_zero=True,
) -> float:
    assert min_val <= max_val
    if min_val == max_val:
        return min_val

    start_val, end_val = min_val, max_val
    if descending:
        start_val, end_val = end_val, start_val

    if current >= n_rounds:
        return end_val

    num = current + 1 if avoid_zero else current
    denom = n_rounds + 1 if avoid_zero else n_rounds
    if exp:
        return start_val * ((end_val / start_val) ** (num / denom))
    else:
        return start_val + (end_val - start_val) * (num / denom)


def create_figure_axes(cfg):
    num_subplots = cfg.algorithm.num_levels
    ncols = math.ceil(math.sqrt(num_subplots))
    nrows = math.ceil(num_subplots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = axes.flatten() if num_subplots > 1 else [axes]
    return fig, axes[:num_subplots]


def visualize_heatmaps(logger, model_state, target, cfg):
    fig, axes = create_figure_axes(cfg)
    for level, ax in enumerate(axes, 1):
        visualize_clf_heatmap(
            model_state,
            target,
            cfg,
            is_forward=True,
            level=level,
            fig=fig,
            ax=ax,
        )
    logger.update({f"figures/fwd_clf_vis": [wandb.Image(fig)]})
    plt.close(fig)

    fig, axes = create_figure_axes(cfg)
    for level, ax in enumerate(axes, 1):
        visualize_clf_heatmap(
            model_state,
            target,
            is_forward=False,
            level=level,
            fig=fig,
            ax=ax,
        )
    logger.update({f"figures/bwd_clf_vis": [wandb.Image(fig)]})
    plt.close(fig)

    # fig, axes = create_figure_axes(cfg)
    # for level, ax in enumerate(axes, 1):
    #     visualize_flow_clf_heatmap(
    #         model_state,
    #         target,
    #         level=level,
    #         fig=fig,
    #         ax=ax,
    #     )
    # logger.update({f"figures/flow_bwd_clf_vis": [wandb.Image(fig)]})
    # plt.close(fig)

    fig, axes = create_figure_axes(cfg)
    for level, ax in enumerate(axes, 1):
        visualize_flow_heatmap(
            model_state,
            target,
            level=level,
            fig=fig,
            ax=ax,
        )
    logger.update({f"figures/flow_vis": [wandb.Image(fig)]})
    plt.close(fig)


def visualize_level_log_reward(level_log_reward_fn, target, cfg, prefix="", show=False):
    num_levels = cfg.algorithm.num_levels + 1
    if cfg.target.dim != 2:
        return

    import jax
    import jax.numpy as jnp

    bounds = (-target._plot_bound, target._plot_bound)
    grid_size = 100
    x = jnp.linspace(bounds[0], bounds[1], grid_size)
    y = jnp.linspace(bounds[0], bounds[1], grid_size)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    rewards_per_level = []
    for level in range(num_levels):
        l_jax = jnp.array(level)
        log_reward = level_log_reward_fn(grid, l_jax)
        rewards_per_level.append(np.asarray(log_reward).reshape(grid_size, grid_size))

    n_cols = min(num_levels, 4)
    n_rows = (num_levels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
    )
    for idx, ax in enumerate(axes.flat):
        if idx >= num_levels:
            ax.axis("off")
            continue
        im = ax.contourf(
            X,
            Y,
            rewards_per_level[idx],
            levels=20,
        )
        ax.set_title(f"Level {idx}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    wb = {f"figures/{prefix + '_' if prefix else ''}vis": [wandb.Image(fig)]}
    if show:
        plt.show()
    else:
        plt.close()
    return wb
