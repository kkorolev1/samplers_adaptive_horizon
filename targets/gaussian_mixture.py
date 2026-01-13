from typing import Callable
import chex
import distrax
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import wandb
from matplotlib import pyplot as plt
from scipy.stats import wishart

from utils.plot_utils import plot_contours_2D, plot_marginal_pair
from targets.base_target import Target


class GaussianMixtureModel(Target):
    def __init__(self, num_components, dim, log_Z=0.0, can_sample=True, sample_bounds=None) -> None:
        # parameters
        super().__init__(dim, log_Z, can_sample)

        self.num_components = num_components

        # parameters
        self.min_mean_val = -10
        self.max_mean_val = 10
        min_val_mixture_weight = 0.3
        max_val_mixture_weight = 0.7
        degree_of_freedom_wishart = dim + 2

        seed = jax.random.PRNGKey(0)

        # set mixture components
        self.locs = jax.random.uniform(
            seed, minval=self.min_mean_val, maxval=self.max_mean_val, shape=(num_components, dim)
        )
        self.covariances = []
        for _ in range(num_components):
            seed, subkey = random.split(seed)

            # Set the random seed for Scipy
            seed_value = random.randint(key=subkey, shape=(), minval=0, maxval=2**30)
            np.random.seed(seed_value)

            cov_matrix = wishart.rvs(df=degree_of_freedom_wishart, scale=jnp.eye(dim))
            self.covariances.append(cov_matrix)
        self.covariances = jnp.array(self.covariances)

        component_dist = distrax.MultivariateNormalFullCovariance(self.locs, self.covariances)

        # set mixture weights
        uniform_mws = True
        if uniform_mws:
            self.mixture_weights = distrax.Categorical(
                logits=jnp.ones(num_components) / num_components
            )
        else:
            self.mixture_weights = distrax.Categorical(
                logits=dist.Uniform(low=min_val_mixture_weight, high=max_val_mixture_weight).sample(
                    seed, sample_shape=(num_components,)
                )
            )

        self.mixture_distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_weights, components_distribution=component_dist
        )

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return self.mixture_distribution.sample(seed=seed, sample_shape=sample_shape)

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        log_prob = self.mixture_distribution.log_prob(x)
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
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

        components_dist = distrax.MultivariateNormalFullCovariance(
            jnp.sqrt(1 - lambda_t) * self.locs,
            (1 - lambda_t) * self.covariances + init_std**2 * lambda_t * jnp.eye(self.dim),
        )
        t_marginal_distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_weights,
            components_distribution=components_dist,
        )

        log_prob = t_marginal_distribution.log_prob(x)
        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)
        return log_prob

    def entropy(self, samples: chex.Array = None):
        expanded = jnp.expand_dims(samples, axis=-2)
        # Compute `log_prob` in every component.
        idx = jnp.argmax(self.mixture_distribution.components_distribution.log_prob(expanded), 1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_components)))
        return entropy

    def visualise(
        self,
        samples: chex.Array | None = None,
        show=False,
        prefix="",
        log_prob_fn: Callable[[chex.Array], chex.Array] | None = None,
    ) -> dict:
        if self.dim == 2:
            bounds = (self.min_mean_val * 1.8, self.max_mean_val * 1.8)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot()
            marginal_dims = (0, 1)
            log_prob_fn = log_prob_fn or self.log_prob
            plot_contours_2D(
                log_prob_fn, self.dim, ax, marginal_dims=marginal_dims, bounds=bounds, levels=50
            )
            if samples is not None:
                plot_marginal_pair(
                    samples[:, marginal_dims], ax, marginal_dims=marginal_dims, bounds=bounds
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
            return {}


if __name__ == "__main__":
    key = jax.random.PRNGKey(45)
    gmm = GaussianMixtureModel(dim=2)
    samples = gmm.sample(key, (1000,))
    print(gmm.entropy(samples))
    # sample = gmm.sample(key, (1,))
    # print(sample)
    # print(samples)
    # print((gmm.log_prob(sample)).shape)
    # print((jax.vmap(gmm.log_prob)(sample)).shape)
    gmm.visualise(show=True)
