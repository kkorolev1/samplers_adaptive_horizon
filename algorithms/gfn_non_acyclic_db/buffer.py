from functools import partial
from typing import Literal, NamedTuple, Protocol, Tuple

import chex
import jax
import jax.numpy as jnp

from algorithms.gfn_tb.sampling_utils import binary_search_smoothing, get_sampling_func


### Helper Functions ###


def get_priorities(
    prioritize_by: str,
    log_iws: chex.Array,
    log_rewards: chex.Array,
    losses: chex.Array,
    target_ess: float = 0.0,
) -> chex.Array:
    batch_size = log_iws.shape[0]  # type: ignore
    log_iws = jax.lax.stop_gradient(log_iws)
    log_rewards = jax.lax.stop_gradient(log_rewards)
    losses = jax.lax.stop_gradient(losses)
    match prioritize_by:
        case "none":
            return jnp.ones((batch_size,))
        case "reward":
            return log_rewards
        case "loss":  # TB loss
            return jnp.log(losses)
        case "uiw":
            if target_ess > 0.0:
                log_iws, _ = binary_search_smoothing(log_iws, target_ess)
            log_iws = jax.nn.log_softmax(log_iws, axis=0)
            return log_iws
        case "piw":
            return log_iws  # Will be smoothed in the `sample` function
        case _:
            raise ValueError(f"Invalid prioritize_by: {prioritize_by}")


### Core Data Structures ###


class TerminalStateData(NamedTuple):
    """
    Holds the core arrays for the terminal state buffer.

    Attributes:
        states: The terminal states. Shape is `(max_length, dim)`.
        priorities: The priorities used for prioritized sampling. Shape is `(max_length,)`.
    """

    states: chex.Array
    log_rewards: chex.Array
    priorities: chex.Array


class TerminalStateBufferState(NamedTuple):
    """
    Represents the complete state of the buffer at any point in time.

    Attributes:
        data: An instance of TerminalStateData holding the arrays.
        current_index: The index for the next insertion in the circular buffer.
        is_full: A boolean flag, True if the buffer has been filled at least once.
    """

    data: TerminalStateData
    current_index: jnp.int32  # type: ignore
    is_full: jnp.bool_  # type: ignore


### Protocol Definitions for Buffer API ###


class InitFn(Protocol):
    def __call__(
        self,
        dtype: jnp.dtype,
        device: jax.Device,  # type: ignore
    ) -> TerminalStateBufferState:
        """Initialises the buffer state with a starting batch of data."""
        ...


class AddFn(Protocol):
    def __call__(
        self,
        buffer_state: TerminalStateBufferState,
        states: chex.Array,
        log_iws: chex.Array,
        log_rewards: chex.Array,
        losses: chex.Array,
    ) -> TerminalStateBufferState:
        """Adds a new batch of data to the buffer."""
        ...


class SampleFn(Protocol):
    def __call__(
        self,
        buffer_state: TerminalStateBufferState,
        key: chex.PRNGKey,
        batch_size: int,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Samples a batch from the buffer.

        Returns:
            A tuple of (sampled_states, sampled_indices).
        """
        ...


class UpdatePriorityFn(Protocol):
    def __call__(
        self,
        buffer_state: TerminalStateBufferState,
        indices: chex.Array,
        log_iws: chex.Array,
        log_rewards: chex.Array,
        losses: chex.Array,
    ) -> TerminalStateBufferState:
        """
        Updates the priorities for a given set of indices.
        This can be used to update priorities or to discard items by setting priorities to -inf.
        """
        ...


class TerminalBuffer(NamedTuple):
    """
    A container for the buffer API functions.

    Attributes:
        init: Function to initialize the buffer.
        add: Function to add new data.
        sample: Function to sample data.
        update_priority: Function to adjust priorities.
        max_length: The maximum capacity of the buffer.
    """

    init: InitFn
    add: AddFn
    sample: SampleFn
    update_priority: UpdatePriorityFn
    max_length: int


### Build Buffer ###


def build_terminal_state_buffer(
    dim: int,
    max_length: int,
    prioritize_by: str,
    target_ess: float = 0.0,
    sampling_method: Literal["multinomial", "stratified", "systematic", "rank"] = "multinomial",
    rank_k: float = 0.01,
) -> TerminalBuffer:
    """
    Creates a prioritized replay buffer for terminal states using a circular buffer.

    Args:
        dim: The dimension of the states to be stored.
        max_length: The maximum capacity of the buffer.
        prioritize_by: The method to use for prioritization.
        target_ess: The target ESS for smoothing.
        sampling_method: The method to use for sampling.
        rank_k: The rank parameter for rank-based sampling.
    """
    assert max_length > 0, "max_length must be greater than 0."
    assert sampling_method in [
        "multinomial",
        "stratified",
        "systematic",
        "rank",
    ], "Invalid sampling method."

    get_priorities_partial = partial(
        get_priorities, prioritize_by=prioritize_by, target_ess=target_ess
    )
    sampling_func = get_sampling_func(sampling_method, rank_k=rank_k)
    sample_with_replacement = sampling_method != "rank"

    def init(
        dtype: jnp.dtype = jnp.float32,
        device: jax.Device = jax.devices("cpu")[0],  # type: ignore
    ) -> TerminalStateBufferState:
        """Initialises the buffer state with a starting batch of data."""

        # Pre-allocate memory for the buffer
        buffer_states = jnp.zeros((max_length, dim), dtype=dtype, device=device)
        buffer_log_rewards = jnp.zeros((max_length,), dtype=dtype, device=device)
        buffer_priorities = -jnp.inf * jnp.ones((max_length,), dtype=dtype, device=device)

        data = TerminalStateData(
            states=buffer_states,
            log_rewards=buffer_log_rewards,
            priorities=buffer_priorities,
        )

        # Create an initial empty state
        buffer_state = TerminalStateBufferState(
            data=data,
            current_index=jnp.int32(0),
            is_full=jnp.bool_(False),
        )

        # Add the initial data
        return buffer_state

    def add(
        buffer_state: TerminalStateBufferState,
        states: chex.Array,
        log_iws: chex.Array,
        log_rewards: chex.Array,
        losses: chex.Array,
    ) -> TerminalStateBufferState:
        """Adds a new batch of data to the buffer."""
        chex.assert_rank(states, 2)
        priorities = get_priorities_partial(log_iws=log_iws, log_rewards=log_rewards, losses=losses)
        chex.assert_rank(priorities, 1)
        batch_size = states.shape[0]  # type: ignore

        # Calculate insertion indices for the circular buffer
        indices = (jnp.arange(batch_size) + buffer_state.current_index) % max_length

        # Update data arrays immutably
        new_states_array = buffer_state.data.states.at[indices].set(states)  # type: ignore
        new_log_rewards_array = buffer_state.data.log_rewards.at[indices].set(log_rewards)  # type: ignore
        new_priorities_array = buffer_state.data.priorities.at[indices].set(priorities)  # type: ignore
        data = TerminalStateData(
            states=new_states_array,
            log_rewards=new_log_rewards_array,
            priorities=new_priorities_array,
        )

        # Update metadata
        new_index = buffer_state.current_index + batch_size
        is_full = jax.lax.select(
            buffer_state.is_full, buffer_state.is_full, new_index >= max_length
        )
        current_index = new_index % max_length

        return TerminalStateBufferState(
            data=data,
            current_index=current_index,
            is_full=is_full,
        )

    def sample(
        buffer_state: TerminalStateBufferState, key: chex.PRNGKey, batch_size: int
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Samples a batch from the buffer in proportion to priorities."""
        # Determine the number of valid items currently in the buffer
        buffer_size = jax.lax.select(buffer_state.is_full, max_length, buffer_state.current_index)

        # Get priorities of valid items
        valid_priorities = buffer_state.data.priorities[:buffer_size]  # type: ignore

        if prioritize_by == "piw":
            valid_priorities, _ = binary_search_smoothing(valid_priorities, target_ess)
        weights = jax.nn.softmax(valid_priorities, axis=0)

        # Sample indices based on the calculated logits
        indices = sampling_func(key, weights, batch_size, sample_with_replacement)

        # Gather the data using the sampled indices
        sampled_states = buffer_state.data.states[indices]  # type: ignore
        sampled_log_rewards = buffer_state.data.log_rewards[indices]  # type: ignore
        return sampled_states, sampled_log_rewards, indices

    def update_priority(
        buffer_state: TerminalStateBufferState,
        indices: chex.Array,
        log_iws: chex.Array,
        log_rewards: chex.Array,
        losses: chex.Array,
    ) -> TerminalStateBufferState:
        """Updates the priorities for a given set of indices."""
        new_priorities = get_priorities_partial(
            log_iws=log_iws, log_rewards=log_rewards, losses=losses
        )
        chex.assert_equal_shape((new_priorities, indices))

        # Update the priorities array immutably and stop gradient flow
        updated_priorities = buffer_state.data.priorities.at[indices].set(  # type: ignore
            jax.lax.stop_gradient(new_priorities)
        )

        # Create new data and state objects
        data = buffer_state.data._replace(priorities=updated_priorities)
        return buffer_state._replace(data=data)

    return TerminalBuffer(
        init=init,
        add=add,
        sample=sample,
        update_priority=update_priority,
        max_length=max_length,
    )
