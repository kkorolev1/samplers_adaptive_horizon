import jax.numpy as jnp
from flax import linen as nn


class NonAcyclicNetML(nn.Module):
    dim: int

    num_layers: int = 3
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2
    use_lp: bool = True

    weight_init: float = 1e-8
    bias_init: float = 0.1

    gamma: float = 1.0
    num_levels: int = 1
    fwd_log_var_range: float = 4.0
    bwd_log_var_range: float = 4.0

    def setup(self):
        self.level_phase = self.param(
            "level_phase", nn.initializers.zeros_init(), (1, self.num_hid)
        )
        self.level_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.level_coder_state = nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu, nn.Dense(self.num_hid)]
        )

        self.fwd_pred_dim = 1 + 2 * self.dim + (self.dim if self.use_lp else 0)
        self.bwd_pred_dim = 1 + 2 * self.dim

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

    def get_fourier_features(self, levels):
        sin_embed_cond = jnp.sin((self.level_coeff * levels) + self.level_phase)
        cos_embed_cond = jnp.cos((self.level_coeff * levels) + self.level_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def _parse_fwd_pred(
        self,
        s,
        model_output,
        lgv_term,
        force_stop=False,
    ):
        if lgv_term is None:
            lgv_term = jnp.zeros_like(s)
        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        if self.use_lp:
            (
                fwd_clf_logits,
                fwd_drift,
                fwd_lgv_scale,
                fwd_scale_corr,
            ) = jnp.split(
                model_output,
                [1, 1 + self.dim, 1 + 2 * self.dim],
                axis=-1,
            )
            fwd_drift = fwd_drift + (1 + fwd_lgv_scale) * lgv_term
        else:
            (
                fwd_clf_logits,
                fwd_drift,
                fwd_scale_corr,
            ) = jnp.split(
                model_output,
                [1, 1 + self.dim],
                axis=-1,
            )
        # fmt: off
        fwd_drift = jnp.clip(fwd_drift, -self.outer_clip, self.outer_clip)
        fwd_mean = s + fwd_drift * self.gamma
        fwd_scale = jnp.sqrt(
            jnp.exp(self.fwd_log_var_range * nn.tanh(fwd_scale_corr)) * self.gamma
        )

        fwd_mean = jnp.clip(fwd_mean, -self.outer_clip, self.outer_clip)
        fwd_clf_logits = fwd_clf_logits.squeeze(-1)

        # if force_stop:
        #     fwd_clf_logits = jnp.full_like(fwd_clf_logits, 100.0)

        return fwd_clf_logits, fwd_mean, fwd_scale

    def _parse_bwd_pred(self, s, model_output, force_stop=False):
        (
            bwd_clf_logits,
            bwd_mean_corr,
            bwd_scale_corr,
        ) = jnp.split(
            model_output,
            [1, 1 + self.dim],
            axis=-1,
        )
        # fmt: off
        bwd_drift = jnp.clip(-nn.softplus(bwd_mean_corr) * s, -self.outer_clip, self.outer_clip)
        bwd_mean = s + bwd_drift * self.gamma
        bwd_scale = jnp.sqrt(
            jnp.exp(self.bwd_log_var_range * nn.tanh(bwd_scale_corr)) * self.gamma
        )
        bwd_mean = jnp.clip(bwd_mean, -self.outer_clip, self.outer_clip)
        bwd_clf_logits = bwd_clf_logits.squeeze(-1)

        # if force_stop:
        #     bwd_clf_logits = jnp.full_like(bwd_clf_logits, 100.0)

        return bwd_clf_logits, bwd_mean, bwd_scale

    def __call__(
        self,
        s,
        l,
        log_reward=None,
        lgv_term=None,
        predict_fwd=True,
        force_stop=False,
    ):
        l = l.astype(jnp.float32) / self.num_levels
        level_array_emb = self.get_fourier_features(l)
        if len(s.shape) == 1:
            level_array_emb = level_array_emb[0]
        l_net1 = self.level_coder_state(level_array_emb)
        s_ext = jnp.concatenate((s, l_net1), axis=-1)

        if predict_fwd:
            model_output = self.fwd_state_net(self.backbone(s_ext))
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
        else:
            model_output = self.bwd_state_net(self.backbone(s_ext))
            return self._parse_bwd_pred(s, model_output, force_stop)
