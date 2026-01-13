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
        sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
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
            nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)
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
            + [nn.Sequential([nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)]
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


class PISGRADNet(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.1

    use_lp: bool = True
    learn_flow: bool = False  # For DB and SubTB
    share_embeddings: bool = False
    flow_num_layers: int = 2
    flow_num_hid: int = 64

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
                + [nn.Sequential([nn.gelu, nn.Dense(self.num_hid)]) for _ in range(self.num_layers)]
                + [nn.gelu]
                + [
                    nn.Dense(
                        self.dim,
                        kernel_init=nn.initializers.constant(self.weight_init),
                        bias_init=nn.initializers.constant(self.bias_init),
                    )
                ]
            )

        self.state_time_net = nn.Sequential(
            [nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)]
            + [
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(1e-8),
                    bias_init=nn.initializers.zeros_init(),
                )
            ]
        )

        self.flow_state_time_net = None
        if self.learn_flow:
            self.flow_timestep_phase = None
            self.flow_time_coder_state = None
            if not self.share_embeddings:
                self.flow_timestep_phase = self.param(
                    "flow_timestep_phase", nn.initializers.zeros_init(), (1, self.flow_num_hid)
                )
                self.flow_timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.flow_num_hid)[
                    None
                ]
                self.flow_time_coder_state = nn.Sequential(
                    [nn.Dense(self.flow_num_hid), nn.gelu, nn.Dense(self.flow_num_hid)]
                )

            self.flow_state_time_net = nn.Sequential(
                [
                    nn.Sequential([nn.Dense(self.flow_num_hid), nn.gelu])
                    for _ in range(self.flow_num_layers)
                ]
                + [
                    nn.Dense(
                        1,
                        kernel_init=nn.initializers.constant(1e-8),
                        bias_init=nn.initializers.zeros_init(),
                    )
                ]
            )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def get_flow_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin((self.flow_timestep_coeff * timesteps) + self.flow_timestep_phase)
        cos_embed_cond = jnp.cos((self.flow_timestep_coeff * timesteps) + self.flow_timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, time_array, lgv_term):
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]
        t_net1 = self.time_coder_state(time_array_emb)
        extended_input = jnp.concatenate((input_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)

        if self.use_lp:
            assert self.time_coder_grad is not None
            t_net2 = self.time_coder_grad(time_array_emb)
            lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
            out_state = out_state + t_net2 * lgv_term

        log_flow = jnp.array(0.0)
        if self.learn_flow:
            assert self.flow_state_time_net is not None
            if not self.share_embeddings:
                assert self.flow_timestep_phase is not None
                assert self.flow_time_coder_state is not None
                flow_time_array_emb = self.get_flow_fourier_features(time_array)
                if len(input_array.shape) == 1:
                    flow_time_array_emb = flow_time_array_emb[0]
                flow_t_net1 = self.flow_time_coder_state(flow_time_array_emb)
                flow_extended_input = jnp.concatenate((input_array, flow_t_net1), axis=-1)
            else:
                flow_extended_input = extended_input
            log_flow = self.flow_state_time_net(flow_extended_input).squeeze(-1)

        return out_state, log_flow
