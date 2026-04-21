from typing import List

import chex
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb

from targets.base_target import Target


class Banana(Target):
    """
    JAX version of the "Banana" energy from the provided Torch implementation.

    Unnormalized log density:
      U(x) = x0^2 / (2p) + 1/2 * (x1 + b x0^2 - b p)^2 + 1/2 * sum_{i>=2} x_i^2
      log_prob(x) = -U(x)

    Sampling uses the standard reparameterization used in the Torch code:
      x ~ N(0, diag([p, 1, 1, ...]))
      x1 <- x1 - b x0^2 + b p
    """

    def __init__(
        self,
        dim: int = 2,
        p: float = 20.0,
        b: float = 0.1,
        can_sample: bool = True,
        sample_bounds=None,
    ) -> None:
        if dim < 2:
            raise ValueError("Not enough dimensions (need dim >= 2).")

        self.data_ndim = dim
        self.p = p
        self.b = b
        self.sample_bounds = sample_bounds
        self._plot_bound = 15.0

        # Normalizing constant of N(0, diag([p, 1, ..., 1])) (matches Torch code).
        log_Z = (dim / 2.0) * jnp.log(2.0 * jnp.pi) + 0.5 * jnp.log(p)
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)

        var = jnp.ones((dim,), dtype=jnp.float32).at[0].set(p)
        self.base_dist = distrax.MultivariateNormalDiag(
            loc=jnp.zeros((dim,), dtype=jnp.float32),
            scale_diag=jnp.sqrt(var),
        )

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        energy = (
            x[..., 0] ** 2 / (2.0 * self.p)
            + 0.5 * (x[..., 1] + self.b * x[..., 0] ** 2 - self.p * self.b) ** 2
            + 0.5 * (x[..., 2:] ** 2).sum(axis=-1)
        )
        logp = -energy

        if not batched:
            logp = jnp.squeeze(logp, axis=0)
        return logp

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        samples = self.base_dist.sample(seed=seed, sample_shape=sample_shape)

        # Banana transform (matches Torch code).
        x0 = samples[..., 0]
        x1 = samples[..., 1] - self.b * x0**2 + self.p * self.b
        samples = samples.at[..., 1].set(x1)

        if self.sample_bounds is not None:
            samples = samples.clip(min=self.sample_bounds[0], max=self.sample_bounds[1])
        return samples

    def visualise(
        self,
        samples: chex.Array = None,
        axes: List[plt.Axes] = None,
        show: bool = False,
        prefix: str = "",
    ) -> dict:
        plt.close()

        if self.dim != 2:
            return {}

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()

        bound = float(self._plot_bound)
        xg, yg = jnp.meshgrid(
            jnp.linspace(-bound, bound, 200), jnp.linspace(-bound, bound, 200)
        )
        grid = jnp.c_[xg.ravel(), yg.ravel()]
        pdf_values = jnp.exp(jax.vmap(self.log_prob)(grid))
        pdf_values = jnp.reshape(pdf_values, xg.shape)
        ax.contourf(xg, yg, pdf_values, levels=30, cmap="viridis")

        if samples is not None:
            idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
            ax.scatter(samples[idx, 0], samples[idx, 1], c="r", alpha=0.5, marker="x")

        ax.set_xticks([])
        ax.set_yticks([])

        tag = f"{prefix + '_' if prefix else ''}vis"
        wb = {f"figures/{tag}": [wandb.Image(fig)]}
        if show:
            plt.show()
        else:
            plt.close()
        return wb
