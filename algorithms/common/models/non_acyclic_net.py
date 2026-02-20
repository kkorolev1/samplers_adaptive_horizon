import jax.numpy as jnp
from flax import linen as nn


class NonAcyclicNet(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.1

    gamma: float = 1.0
    fwd_log_var_range: float = 4.0
    bwd_log_var_range: float = 4.0
    learn_fwd_corrections: bool = False
    shared_model: bool = False
    disable_clf: bool = False

    def setup(self):
        self.fwd_pred_dim = 1 + 3 * self.dim if self.learn_fwd_corrections else 1
        self.bwd_pred_dim = 1 + 2 * self.dim
        if self.shared_model:
            self.state_net = nn.Sequential(
                [
                    nn.Sequential([nn.Dense(self.num_hid), nn.gelu])
                    for _ in range(self.num_layers)
                ]
                + [
                    nn.Dense(
                        self.fwd_pred_dim + self.bwd_pred_dim,
                        kernel_init=nn.initializers.constant(1e-8),
                        bias_init=nn.initializers.zeros_init(),
                    )
                ]
            )
        else:
            self.fwd_state_net = nn.Sequential(
                [
                    nn.Sequential([nn.Dense(self.num_hid), nn.gelu])
                    for _ in range(self.num_layers)
                ]
                + [
                    nn.Dense(
                        self.fwd_pred_dim,
                        kernel_init=nn.initializers.constant(1e-8),
                        bias_init=nn.initializers.zeros_init(),
                    )
                ]
            )
            self.bwd_state_net = nn.Sequential(
                [
                    nn.Sequential([nn.Dense(self.num_hid), nn.gelu])
                    for _ in range(self.num_layers)
                ]
                + [
                    nn.Dense(
                        self.bwd_pred_dim,
                        kernel_init=nn.initializers.constant(1e-8),
                        bias_init=nn.initializers.zeros_init(),
                    )
                ]
            )

    def _parse_fwd_pred(
        self,
        s,
        model_output,
        lgv_term,
        force_stop=False,
    ):
        if self.shared_model:
            model_output, _ = jnp.split(model_output, [self.fwd_pred_dim], axis=-1)
        if self.learn_fwd_corrections:
            (
                fwd_clf_logits,
                fwd_mean_corr,
                fwd_lgv_scale,
                fwd_scale_corr,
            ) = jnp.split(
                model_output,
                [1, 1 + self.dim, 1 + 2 * self.dim],
                axis=-1,
            )
            # fmt: off
            fwd_mean = s + (fwd_mean_corr + (1 + fwd_lgv_scale) * lgv_term) * self.gamma
            fwd_scale = jnp.sqrt(
                2 * jnp.exp(self.fwd_log_var_range * nn.tanh(fwd_scale_corr)) * self.gamma
            )
        else:
            fwd_clf_logits = model_output
            fwd_mean = s + lgv_term * self.gamma
            fwd_scale = jnp.sqrt(2 * self.gamma)

        fwd_mean = jnp.clip(fwd_mean, -self.outer_clip, self.outer_clip)
        fwd_clf_logits = fwd_clf_logits.squeeze(-1)

        if self.disable_clf:
            fwd_clf_logits = jnp.full_like(fwd_clf_logits, -100.0)

        if force_stop:
            fwd_clf_logits = jnp.full_like(fwd_clf_logits, 100.0)

        return fwd_clf_logits, fwd_mean, fwd_scale

    def _parse_bwd_pred(self, s, model_output, force_stop=False):
        if self.shared_model:
            _, model_output = jnp.split(model_output, [self.fwd_pred_dim], axis=-1)

        (
            bwd_clf_logits,
            bwd_mean_corr,
            bwd_scale_corr,
        ) = jnp.split(
            model_output,
            [1, 1 + self.dim],
            axis=-1,
        )
        bwd_mean = s - nn.softplus(bwd_mean_corr) * s * self.gamma
        bwd_mean = jnp.clip(bwd_mean, -self.outer_clip, self.outer_clip)
        bwd_scale = jnp.sqrt(
            jnp.exp(self.bwd_log_var_range * nn.tanh(bwd_scale_corr)) * self.gamma
        )
        bwd_clf_logits = bwd_clf_logits.squeeze(-1)

        if self.disable_clf:
            bwd_clf_logits = jnp.full_like(bwd_clf_logits, -100.0)

        if force_stop:
            bwd_clf_logits = jnp.full_like(bwd_clf_logits, 100.0)

        return bwd_clf_logits, bwd_mean, bwd_scale

    def __call__(
        self,
        s,
        log_reward=None,
        lgv_term=None,
        predict_fwd=False,
        predict_bwd=False,
        force_stop=False,
    ):
        if predict_fwd:
            model_output = (
                self.state_net(s) if self.shared_model else self.fwd_state_net(s)
            )
            if lgv_term is None:
                lgv_term = jnp.zeros_like(s)
            lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
            fwd_clf_logits, fwd_mean, fwd_scale = self._parse_fwd_pred(
                s,
                model_output,
                lgv_term,
                force_stop,
            )
            if log_reward is None:
                log_flow = jnp.zeros_like(s[..., 0])
            else:
                log_flow = log_reward - nn.log_sigmoid(fwd_clf_logits)
            return fwd_clf_logits, fwd_mean, fwd_scale, log_flow
        if predict_bwd:
            model_output = (
                self.state_net(s) if self.shared_model else self.bwd_state_net(s)
            )
            return self._parse_bwd_pred(s, model_output, force_stop)
