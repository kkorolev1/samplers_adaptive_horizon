import jax.numpy as jnp
from flax import linen as nn


class StateTimeEncoder(nn.Module):
    num_layers: int = 2
    num_hid: int = 64
    zero_init: bool = False

    def setup(self):
        if self.zero_init:
            last_layer = [
                nn.Dense(
                    self.parent.dim,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                )
            ]
        else:
            # last_layer = [nn.Dense(self.parent.dim)]
            last_layer = [
                nn.Dense(
                    self.parent.dim,
                    kernel_init=nn.initializers.normal(stddev=1e-7),
                    bias_init=nn.initializers.zeros_init(),
                )
            ]

        self.state_time_net = [
            nn.Sequential([nn.Dense(self.num_hid), nn.gelu])
            for _ in range(self.num_layers)
        ] + last_layer

    def __call__(self, extended_input):
        for layer in self.state_time_net:
            extended_input = layer(extended_input)
        return extended_input


class LangevinScaleNet(nn.Module):
    num_layers: int = 2
    num_hid: int = 64
    lgv_per_dim: bool = False

    def setup(self):
        self.time_coder_grad = (
            [nn.Dense(self.num_hid)]
            + [
                nn.Sequential([nn.gelu, nn.Dense(self.num_hid)])
                for _ in range(self.num_layers)
            ]
            + [
                nn.gelu,
                nn.Dense(
                    self.parent.dim if self.lgv_per_dim else 1,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                ),
            ]
        )

    def __call__(self, time_array_emb):
        for layer in self.time_coder_grad:
            time_array_emb = layer(time_array_emb)
        return time_array_emb


class NonAcyclicNet(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.1

    step_size: float = 1.0
    fwd_log_var_range: float = 4.0
    bwd_log_var_range: float = 4.0
    learn_fwd: bool = False

    def setup(self):
        self.fwd_pred_dim = 1 + 3 * self.dim if self.learn_fwd else 1
        self.bwd_pred_dim = 1 + 2 * self.dim
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

    def __call__(self, input_array, lgv_term=None, log_reward=None):
        model_output = self.state_net(input_array)
        d = input_array.shape[-1]

        # if we don't want to calculate fwd
        if lgv_term is None:
            lgv_term = jnp.zeros_like(input_array)

        if self.learn_fwd:
            (
                fwd_clf_logits,
                fwd_mean_corr,
                fwd_lgv_scale,
                fwd_scale_corr,
                bwd_clf_logits,
                bwd_mean_corr,
                bwd_scale_corr,
            ) = jnp.split(
                model_output,
                [1, 1 + d, 1 + 2 * d, 1 + 3 * d, 2 + 3 * d, 2 + 4 * d],
                axis=-1,
            )
            # fmt: off
            fwd_mean = input_array + (fwd_mean_corr + fwd_lgv_scale * lgv_term) * self.step_size
            fwd_scale = jnp.sqrt(
                2 * jnp.exp(self.fwd_log_var_range * nn.tanh(fwd_scale_corr)) * self.step_size
            )
        else:
            fwd_clf_logits, bwd_clf_logits, bwd_mean_corr, bwd_scale_corr = jnp.split(
                model_output, [1, 2, 2 + d], axis=-1
            )
            fwd_mean = input_array + lgv_term * self.step_size
            fwd_scale = jnp.sqrt(2 * self.step_size)

        fwd_clf_logits = fwd_clf_logits.squeeze(-1)
        bwd_clf_logits = bwd_clf_logits.squeeze(-1)

        # DEBUG:
        bwd_mean_corr = jnp.array(0.0)
        bwd_scale_corr = jnp.array(0.0)

        # fmt: off
        bwd_mean = input_array - nn.softplus(bwd_mean_corr) * input_array * self.step_size
        bwd_scale = jnp.sqrt(
            jnp.exp(self.bwd_log_var_range * nn.tanh(bwd_scale_corr)) * self.step_size
        )

        log_flow = jnp.array(0.0)
        # if we want to calculate log flow
        if log_reward is not None:
            log_flow = log_reward - nn.log_sigmoid(fwd_clf_logits)
        return (
            (fwd_clf_logits, fwd_mean, fwd_scale),
            (bwd_clf_logits, bwd_mean, bwd_scale),
            log_flow,
        )
