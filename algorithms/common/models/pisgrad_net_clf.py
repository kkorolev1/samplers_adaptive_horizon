import jax.numpy as jnp
from flax import linen as nn


class TimeEncoder(nn.Module):
    num_hid: int = 2

    def setup(self):
        self.timestep_phase = self.param(
            "timestep_phase", nn.initializers.zeros_init(), (1, self.num_hid)
        )
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.mlp = [
            nn.Dense(2 * self.num_hid),
            nn.gelu,
            nn.Dense(self.num_hid),
        ]

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, time_array_emb):
        time_array_emb = self.get_fourier_features(time_array_emb)
        for layer in self.mlp:
            time_array_emb = layer(time_array_emb)
        return time_array_emb


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


class PISGRADNetClf(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.1

    use_lp: bool = True
    base_var: float = 1.0
    num_steps: int = 100
    shared_model: bool = False

    fwd_scale_scalar: float = 4.0
    bwd_mean_scalar: float = 0.9
    bwd_scale_scalar: float = 0.9

    def setup(self):
        self.timestep_phase = self.param(
            "timestep_phase", nn.initializers.zeros_init(), (1, self.num_hid)
        )
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu, nn.Dense(self.num_hid)]
        )

        self.time_coder_grad = None
        if self.use_lp:
            self.time_coder_grad = nn.Sequential(
                [nn.Dense(self.num_hid)]
                + [
                    nn.Sequential([nn.gelu, nn.Dense(self.num_hid)])
                    for _ in range(self.num_layers)
                ]
                + [nn.gelu]
                + [
                    nn.Dense(
                        self.dim,
                        kernel_init=nn.initializers.constant(self.weight_init),
                        bias_init=nn.initializers.constant(self.bias_init),
                    )
                ]
            )

        self.fwd_pred_dim = 1 + 2 * self.dim
        self.bwd_pred_dim = 2 * self.dim
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
            self.backbone = nn.Sequential(
                [
                    nn.Sequential([nn.Dense(self.num_hid), nn.gelu])
                    for _ in range(self.num_layers - 1)
                ]
            )

            self.fwd_state_net = nn.Sequential(
                [
                    nn.Dense(self.num_hid),
                    nn.gelu,
                    nn.Dense(
                        self.fwd_pred_dim,
                        kernel_init=nn.initializers.constant(1e-8),
                        bias_init=nn.initializers.zeros_init(),
                    ),
                ]
            )
            self.bwd_state_net = nn.Sequential(
                [
                    nn.Dense(self.num_hid),
                    nn.gelu,
                    nn.Dense(
                        self.bwd_pred_dim,
                        kernel_init=nn.initializers.constant(1e-8),
                        bias_init=nn.initializers.zeros_init(),
                    ),
                ]
            )
        # self.step_net = nn.Sequential(
        #     [
        #         nn.Dense(self.num_hid),
        #         nn.gelu,
        #         nn.Dense(
        #             self.num_steps,
        #             kernel_init=nn.initializers.constant(1e-8),
        #             bias_init=nn.initializers.zeros_init(),
        #         ),
        #     ]
        # )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def _parse_fwd_pred(
        self,
        s,
        time_array_emb,
        model_output,
        lgv_term,
        force_stop=False,
    ):
        if lgv_term is None:
            lgv_term = jnp.zeros_like(s)
        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        if self.shared_model:
            model_output, _ = jnp.split(model_output, [self.fwd_pred_dim], axis=-1)

        (
            fwd_clf_logits,
            fwd_drift,
            fwd_scale_corr,
        ) = jnp.split(
            model_output,
            [1, 1 + self.dim],
            axis=-1,
        )
        if self.use_lp:
            assert self.time_coder_grad is not None
            t_net2 = self.time_coder_grad(time_array_emb)
            lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
            fwd_drift = fwd_drift + t_net2 * lgv_term
        fwd_drift = jnp.clip(fwd_drift, -self.outer_clip, self.outer_clip)
        dt = 1 / self.num_steps
        fwd_mean = s + fwd_drift * dt
        # fmt: off
        fwd_scale = jnp.sqrt(
            jnp.exp(self.fwd_scale_scalar * nn.tanh(fwd_scale_corr)) * self.base_var * dt
        )
        fwd_clf_logits = fwd_clf_logits.squeeze(-1)

        if force_stop:
            fwd_clf_logits = jnp.full_like(fwd_clf_logits, 100.0)

        return fwd_clf_logits, fwd_mean, fwd_scale

    def _parse_bwd_pred(self, s, t, model_output):
        if self.shared_model:
            _, model_output = jnp.split(model_output, [self.fwd_pred_dim], axis=-1)

        (
            bwd_mean_corr,
            bwd_scale_corr,
        ) = jnp.split(
            model_output,
            [self.dim],
            axis=-1,
        )
        dt = 1 / self.num_steps
        bwd_mean_corr = 1 + self.bwd_mean_scalar * nn.tanh(bwd_mean_corr)
        bwd_mean = s - bwd_mean_corr * s / jnp.expand_dims(t, -1) * dt
        bwd_scale_corr = 1 + self.bwd_scale_scalar * nn.tanh(bwd_scale_corr)
        bwd_scale = jnp.sqrt(bwd_scale_corr * (t - dt) / t * self.base_var * dt)

        return bwd_mean, bwd_scale

    def __call__(
        self,
        s,
        t,
        log_reward=None,
        lgv_term=None,
        predict_fwd=True,
        force_stop=False,
    ):
        time_array_emb = self.get_fourier_features(t)
        if len(s.shape) == 1:
            time_array_emb = time_array_emb[0]
        t_net1 = self.time_coder_state(time_array_emb)
        s_ext = jnp.concatenate((s, t_net1), axis=-1)

        if predict_fwd:
            model_output = (
                self.state_net(s_ext)
                if self.shared_model
                else self.fwd_state_net(self.backbone(s_ext))
            )
            fwd_clf_logits, fwd_mean, fwd_scale = self._parse_fwd_pred(
                s,
                time_array_emb,
                model_output,
                lgv_term,
                force_stop,
            )
            if log_reward is None:
                log_flow = jnp.zeros_like(s[..., 0])
            else:
                log_flow = log_reward - nn.log_sigmoid(fwd_clf_logits)
            return fwd_clf_logits, fwd_mean, fwd_scale, log_flow
        else:
            model_output = (
                self.state_net(s_ext)
                if self.shared_model
                else self.bwd_state_net(self.backbone(s_ext))
            )
            return self._parse_bwd_pred(s, t, model_output)
