import os
import pickle
from flax import serialization
import jax
from typing import Tuple

from algorithms.common.diffusion_related.init_model import init_model_non_acyclic

import os
from datetime import datetime
import jax.numpy as jnp
import distrax
import itertools
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

import hydra
import jax
import matplotlib

from omegaconf import DictConfig, OmegaConf
from functools import partial

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from utils.helper import flatten_dict, reset_device_memory
from algorithms.gfn_non_acyclic.gfn_non_acyclic_rnd import rnd_mcmc, rnd_eval
from algorithms.gfn_non_acyclic.gfn_non_acyclic_trainer import _get_checkpoint_path
from utils.plot_utils import (
    plot_gradient_trajectory,
    marginal_density_grid,
    visualize_clf_heatmap,
)


def load_trained_model(cfg, target):
    key = jax.random.PRNGKey(cfg.seed)
    dim = target.dim

    # Recreate the model state with the same architecture/config
    model_state = init_model_non_acyclic(key, dim, cfg.algorithm)
    ckpt_path = _get_checkpoint_path(cfg, iter=cfg.ckpt_iter)

    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)

    params = serialization.from_state_dict(model_state.params, payload["params"])
    model_state = model_state.replace(params=params)

    print(f"Loaded checkpoint from iter {payload.get('iter', 'unknown')}")
    return model_state


white_blue_cmap = LinearSegmentedColormap.from_list(
    "white_blue",
    ["#ffffff", "#dbe9ff", "#8fb6ff", "#3b6fb6", "#0b1f4d"],
)


def visualize_trajectories(
    trajectories,
    trajectories_length,
    target,
    dims=(0, 1),
    num_examples: int = 6,
    show: bool = True,
):
    batch_size = int(trajectories[0].shape[0])
    dim = trajectories[0].shape[-1]
    n = min(int(num_examples), batch_size)
    dim = int(target.dim)
    bounds = (-float(target._plot_bound), float(target._plot_bound))
    # bounds = (-5, 5)
    d0, d1 = int(dims[0]), int(dims[1])

    fig, ax = plt.subplots(figsize=(6, 6))
    x1d, x2d, zd = marginal_density_grid(target, dim, (d0, d1), bounds, n_points=110)

    colors = ["#f7c860", "#ad102d"]

    for i in range(len(trajectories)):
        color = colors[i]
        for traj_index in range(n):
            tl = int(trajectories_length[i][traj_index])
            valid = trajectories[i][traj_index, :tl]
            xy = np.asarray(jnp.stack([valid[:, d0], valid[:, d1]], axis=1))
            plot_gradient_trajectory(ax, xy, color=color)

    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2.5, label="ULA"),
        Line2D([0], [0], color=colors[1], lw=2.5, label="Full model"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        fontsize=10,
    )

    zd = np.maximum(zd, 0.0)
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
    if show:
        plt.show()
    else:
        plt.close(fig)


def eval_fn_one(rnd, model_state, key):
    params = (model_state.params,)
    (
        trajectories,
        _,
        _,
        trajectories_length,
    ) = rnd(key, model_state, *params)
    return trajectories, trajectories_length


def eval_fn(cfg, model_state, target):
    key_gen = jax.random.PRNGKey(12)

    dim = target.dim
    alg_cfg = cfg.algorithm

    initial_dist = distrax.MultivariateNormalDiag(
        jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std
    )

    batch_size = 1
    # cfg.eval_samples

    rnd_ula = partial(
        rnd_mcmc,
        batch_size=batch_size,
        aux_tuple=(alg_cfg.model.gamma,),
        target=target,
        num_steps=alg_cfg.eval_max_steps,
        step_name="ula",
        initial_dist=initial_dist,
    )

    rnd_full_model = partial(
        rnd_eval,
        batch_size=batch_size,
        aux_tuple=(alg_cfg.logr_clip,),
        target=target,
        num_steps=alg_cfg.eval_max_steps,
        use_lp=alg_cfg.model.use_lp,
        initial_dist=initial_dist,
    )

    ula_trajs, ula_traj_lengths = eval_fn_one(rnd_ula, model_state, key_gen)
    trajs, traj_lengths = eval_fn_one(rnd_full_model, model_state, key_gen)
    trajs = [ula_trajs, trajs]
    lengths = [ula_traj_lengths, traj_lengths]
    print("ula lengths: ", jnp.mean(ula_traj_lengths))
    print("full model lengths: ", jnp.mean(traj_lengths))
    # visualize_trajectories(trajs, lengths, target)

    visualize_clf_heatmap(
        model_state, target, cfg, is_forward=True, alpha=0.2, show=True
    )


@hydra.main(version_base=None, config_path="configs", config_name="base_conf")
def main(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Load the chosen algorithm-specific configuration dynamically
    cfg = hydra.utils.instantiate(cfg)
    target = cfg.target.fn

    print("JAX devices:", jax.devices())
    print("JAX default backend:", jax.default_backend())

    if not cfg.visualize_samples:
        matplotlib.use("agg")

    model_state = load_trained_model(cfg, target)

    try:
        if cfg.use_jit:
            eval_fn(cfg, model_state, target)
        else:
            with jax.disable_jit():
                eval_fn(cfg, model_state, target)
    except Exception as e:
        reset_device_memory()
        raise e


if __name__ == "__main__":
    main()
