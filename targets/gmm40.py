from typing import Callable
import chex
import distrax
import jax
import jax.numpy as jnp
import wandb
from matplotlib import pyplot as plt

from utils.plot_utils import plot_contours_2D, plot_marginal_pair
from targets.base_target import Target


class GMM40(Target):
    def __init__(
        self,
        dim: int = 2,
        num_components: int = 40,
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

        key = jax.random.PRNGKey(seed)
        logits = jnp.ones(num_components)
        self.mean = (
            jax.random.uniform(shape=(num_components, dim), key=key, minval=-1.0, maxval=1.0)
            * loc_scaling
        )
        self.scale = jnp.ones(shape=(num_components, dim)) * scale_scaling

        self.mixture_dist = distrax.Categorical(logits=logits)
        components_dist = distrax.Independent(
            distrax.Normal(loc=self.mean, scale=self.scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            components_distribution=components_dist,
        )

        self._plot_bound = loc_scaling * 1.5

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            x = x[None]

        log_prob = self.distribution.log_prob(x)
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
        idx = jnp.argmax(self.distribution.components_distribution.log_prob(expanded), 1)
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
            return {}  # TODO: add visualisation for higher dimensions


if __name__ == "__main__":
    # gmm = GMM40()
    # samples = gmm.sample(jax.random.PRNGKey(0), (2000,))
    # gmm.log_prob(samples)
    # gmm.entropy(samples)
    # # gmm.visualise( show=True)
    # gmm.visualise(show=True)
    from eval import discrepancies

    key = jax.random.PRNGKey(0)
    target = GMM40(dim=50)
    sample1 = target.sample(key, (2000,))

    min_sd = jnp.inf
    max_sd = 0.0
    sd_list = []
    mmd_list = []
    n_trial = 5

    sd_self = discrepancies.compute_sd(sample1, sample1, None)
    print(f"Self sd: {sd_self:.4f}")
    mmd_self = discrepancies.compute_mmd(sample1, sample1, None)
    print(f"Self mmd: {mmd_self:.4f}")

    key = jax.random.PRNGKey(99999)
    _, keygen = jax.random.split(key)
    for i in range(1, n_trial + 1):
        key2, keygen = jax.random.split(keygen)

        sample2 = target.sample(key2, (2000,))
        sd = discrepancies.compute_sd(sample1, sample2, None)
        sd_list.append(sd)
        mmd = discrepancies.compute_mmd(sample1, sample2, None)
        mmd_list.append(mmd)
        if sd < min_sd:
            min_sd = sd
            best_key2 = key2
        if sd > max_sd:
            max_sd = sd
            worst_key2 = key2
        running_mean_sd = sum(sd_list) / i
        running_mean_mmd = sum(mmd_list) / i
        print(
            f"Iteration {i} - Best sd: {min_sd:.2f}, Worst sd: {max_sd:.2f}, Running mean sd: {running_mean_sd:.2f}, Running mean mmd: {running_mean_mmd:.3f}"
        )

    sd_list = jnp.array(sd_list)
    mmd_list = jnp.array(mmd_list)
    mean_sd = sum(sd_list) / n_trial
    std_sd = jnp.std(sd_list)
    mean_mmd = sum(mmd_list) / n_trial
    std_mmd = jnp.std(mmd_list)
    print(
        f"Final (n_trial = {n_trial}) - Best sd: {min_sd:.2f}, Worst sd: {max_sd:.2f}, Mean sd: {mean_sd:.2f}, Std sd: {std_sd:.2f}, Mean mmd: {mean_mmd:.3f}, Std mmd: {std_mmd:.3f}"
    )
