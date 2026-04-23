from typing import Callable
import chex
import distrax
import jax
import jax.numpy as jnp
import wandb
from matplotlib import pyplot as plt

from utils.plot_utils import plot_contours_2D, plot_marginal_pair
from targets.base_target import Target


class GMM9(Target):
    def __init__(
        self,
        dim: int = 2,
        num_components: int = 9,
        loc_scaling: float = 40,
        scale_scaling: float = 1.0,
        seed: int = 0,
        sample_bounds=None,
        can_sample=True,
        log_Z=0,
    ) -> None:
        super().__init__(dim, log_Z, can_sample)

        self.seed = seed
        self.n_mixes = num_components

        if dim == 2:
            values = [-5, 0, 5]
            # override self.mean, self.scale, and mixture
            self.mean = jnp.array(
                [[x, y] for x in values for y in values], dtype=jnp.float32
            )
        else:
            key = jax.random.PRNGKey(0)
            self.mean = jax.random.uniform(
                key,
                shape=(num_components, dim),
                minval=-5.0,
                maxval=5.0,
                dtype=jnp.float32,
            )
        self.scale = jnp.sqrt(jnp.ones(dim, dtype=jnp.float32) * 0.5477222)[
            None, :
        ].repeat(num_components, axis=0)
        weights = jnp.array(
            [0.1277, 0.0553, 0.2590, 0.0973, 0.0416, 0.0241, 0.1662, 0.0185, 0.2102],
            dtype=jnp.float32,
        )
        self.mixture_dist = distrax.Categorical(probs=weights)
        components_dist = distrax.Independent(
            distrax.Normal(loc=self.mean, scale=self.scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = 7

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        log_prob = self.distribution.log_prob(x)
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def log_prob_marginal_pair(self, x_2d: chex.Array, i: int, j: int) -> chex.Array:
        # Marginalize the GMM to coordinates (i, j)
        # Extract relevant (i, j) from means and covariances/scales for each component
        # All weights remain the same; covariance is diagonal

        means_ij = self.mean[:, [i, j]]  # (num_components, 2)
        scales_ij = self.scale[:, [i, j]]  # (num_components, 2)

        mixture_dist = self.mixture_dist
        components_dist = distrax.Independent(
            distrax.Normal(loc=means_ij, scale=scales_ij), reinterpreted_batch_ndims=1
        )

        marginal_distribution = distrax.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            components_distribution=components_dist,
        )

        log_prob = marginal_distribution.log_prob(x_2d)
        return log_prob

    def log_prob_t(
        self,
        x: chex.Array,
        lambda_t: float,  # 1 - exp(-2\int_0^t \beta_s ds)
        init_std: float,  # \sigma
    ) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        components_dist = distrax.Independent(
            distrax.Normal(
                loc=jnp.sqrt(1 - lambda_t) * self.mean,
                scale=jnp.sqrt((1 - lambda_t) * self.scale**2 + init_std**2 * lambda_t),
            ),
            reinterpreted_batch_ndims=1,
        )
        t_marginal_distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            components_distribution=components_dist,
        )

        log_prob = t_marginal_distribution.log_prob(x)
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()) -> chex.Array:
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def entropy(self, samples: chex.Array = None):
        expanded = jnp.expand_dims(samples, axis=-2)
        # Compute `log_prob` in every component.
        idx = jnp.argmax(
            self.distribution.components_distribution.log_prob(expanded), 1
        )
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.n_mixes)))
        return entropy

    def visualise(
        self,
        samples: chex.Array | None = None,
        show=False,
        prefix="",
        log_prob_fn: Callable[[chex.Array], chex.Array] | None = None,
    ) -> dict:
        if self.dim == 2:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot()
            marginal_dims = (0, 1)
            bounds = (-self._plot_bound, self._plot_bound)
            log_prob_fn = log_prob_fn or self.log_prob
            plot_contours_2D(
                log_prob_fn,
                self.dim,
                ax,
                marginal_dims=marginal_dims,
                bounds=bounds,
                levels=50,
            )
            if samples is not None:
                plot_marginal_pair(
                    samples[:, marginal_dims],
                    ax,
                    marginal_dims=marginal_dims,
                    bounds=bounds,
                )
            plt.xticks([])
            plt.yticks([])

            wb = {f"figures/{prefix + '_' if prefix else ''}vis": [wandb.Image(fig)]}
            if show:
                plt.show()
            else:
                plt.close()
            return wb
        else:
            plotting_bounds = (-self._plot_bound, self._plot_bound)
            grid_width_n_points = 100
            fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex="row", sharey="row")
            if samples is not None:
                samples = jnp.clip(samples, plotting_bounds[0], plotting_bounds[1])
            for i in range(2):
                for j in range(2):
                    xx, yy = jnp.meshgrid(
                        jnp.linspace(
                            plotting_bounds[0], plotting_bounds[1], grid_width_n_points
                        ),
                        jnp.linspace(
                            plotting_bounds[0], plotting_bounds[1], grid_width_n_points
                        ),
                    )
                    x_points = jnp.column_stack([xx.ravel(), yy.ravel()])
                    log_probs = self.log_prob_marginal_pair(x_points, i, j + 2)
                    log_probs = jnp.clip(log_probs, -1000, a_max=None).reshape(
                        (grid_width_n_points, grid_width_n_points)
                    )
                    axs[i, j].contour(xx, yy, log_probs, levels=20)

                    if samples is not None:
                        # plot samples
                        axs[i, j].plot(samples[:, i], samples[:, j + 2], "o", alpha=0.5)

                    if j == 0:
                        axs[i, j].set_ylabel(f"$x_{i + 1}$")
                    if i == 1:
                        axs[i, j].set_xlabel(f"$x_{j + 1 + 2}$")

            plt.tight_layout()
            wb = {f"figures/{prefix + '_' if prefix else ''}vis": [wandb.Image(fig)]}
            if show:
                plt.show()
            else:
                plt.close()
            return wb


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    target = GMM9(dim=10)
    samples = target.sample(key, (2000,))

    target.visualise(samples, show=True)
