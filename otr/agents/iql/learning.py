"""Implementations of Implicit Q Learning (IQL) learner component."""
import time
from typing import Any, Dict, Iterator, NamedTuple, Tuple

import acme
from acme import types as core_types
from acme.jax import networks as networks_lib
from acme.jax import types
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax

from otr.agents.iql import networks as iql_networks

_Metrics = Dict[str, jnp.ndarray]


class TrainingState(NamedTuple):
  old_policy_params: networks_lib.Params
  old_policy_opt_state: optax.OptState
  policy_params: networks_lib.Params
  policy_opt_state: optax.OptState
  value_params: networks_lib.Params
  value_opt_state: optax.OptState
  critic_params: networks_lib.Params
  critic_opt_state: optax.OptState
  target_critic_params: networks_lib.Params
  key: types.PRNGKey
  steps: int

class BCTrainingState(NamedTuple):
    
    # Behavior cloning network parameters and optimizer state
  bc_params: networks_lib.Params
  bc_opt_state: optax.OptState
  key: types.PRNGKey


def expectile_loss(diff, expectile=0.8):
  weight = jnp.where(diff > 0, expectile, (1 - expectile))
  return weight * (diff**2)


class IQLLearner(acme.Learner):
  """IQL Learner."""

  _state: TrainingState
  _bc_state: BCTrainingState
  _bc_max_steps: int

  def __init__(
      self,
      random_key: types.PRNGKey,
      random_key_bc: types.PRNGKey,
      networks: iql_networks.IQLNetworks,
      bc_network: networks_lib.FeedForwardNetwork,
      dataset: Iterator[core_types.Transition],
      policy_optimizer: optax.GradientTransformation,
      critic_optimizer: optax.GradientTransformation,
      value_optimizer: optax.GradientTransformation,
      discount: float = 0.99,
      tau: float = 0.005,
      expectile: float = 0.8,
      temperature: float = 0.1,
      max_steps: int = 5e5,
      counter=None,
      logger=None,
  ):
    """Create an instance of the IQLLearner.

        Args:
            random_key (types.PRNGKey): random seed used by the learner.
            networks (iql_networks.IQLNetworks): networks used by the learner.
            dataset (Iterator[core_types.Transition]): dataset iterator.
            policy_optimizer (optax.GradientTransformation): optimizer for policy.
            critic_optimizer (optax.GradientTransformation): optimizer for critic.
            value_optimizer (optax.GradientTransformation): optimizer for value critic.
            discount (float, optional): additional discount. Defaults to 0.99.
            tau (float, optional): target soft update rate. Defaults to 0.005.
            expectile (float, optional): expectile for training critic. Defaults to 0.8.
            temperature (float, optional): temperature for the AWR. Defaults to 0.1.
            counter ([type], optional): counter for keeping counts. Defaults to None.
            logger ([type], optional): logger for writing metrics. Defaults to None.

        Returns:
            An instance of IQLLearner
        """

    policy_network = networks.policy_network
    value_network = networks.value_network
    critic_network = networks.critic_network
    self._bc_max_steps = max_steps

    def bc_loss_fn(
        policy_params: networks_lib.Params,
        key: types.PRNGKey,
        batch: core_types.Transition
        ) -> Tuple[jnp.ndarray, Any]:

      dist = bc_network.apply(
          policy_params, batch.observation, is_training=True, key=key)
      log_probs = dist.log_prob(batch.action)
      bc_loss = -(log_probs).mean()

      return bc_loss, {"bc_loss": bc_loss}




    def awr_actor_loss_fn(
        policy_params: networks_lib.Params,
        old_policy_params: networks_lib.Params,
        key: types.PRNGKey,
        target_critic_params: networks_lib.Params,
        value_params: networks_lib.Params,
        batch: core_types.Transition,
        steps: int,
        clip_ratio: float = .25,
        entropy_weight: float = 0.01,
        is_clip_decay: bool = True
    ) -> Tuple[jnp.ndarray, Any]:
      v = value_network.apply(value_params, batch.observation)
      q1, q2 = critic_network.apply(target_critic_params, batch.observation, batch.action)
      q = jnp.minimum(q1, q2)
      advantage = q - v
      advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

      dist = policy_network.apply(policy_params, batch.observation, is_training=True, key=key)
      log_probs = dist.log_prob(batch.action)
      old_dist = policy_network.apply(old_policy_params, batch.observation, is_training=True, key=key)
      old_log_probs = old_dist.log_prob(batch.action)

      ratio = jnp.exp(log_probs - old_log_probs)

      clip_ratio = jnp.where(steps<200, 0.96*clip_ratio, clip_ratio)

      loss1 = ratio * advantage
      loss2 = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
      entropy_loss = dist.entropy().sum(-1, keepdims=True) * entropy_weight

      actor_loss = -(jnp.minimum(loss1, loss2) + entropy_loss).mean()

      return actor_loss, {
          "actor_loss": actor_loss,
          "advantage": jnp.mean(advantage)
      }


    def value_loss_fn(
        value_params: networks_lib.Params,
        target_critic_params: networks_lib.Params,
        batch: core_types.Transition,
    ) -> Tuple[jnp.ndarray, Any]:
      q1, q2 = critic_network.apply(target_critic_params, batch.observation,
                                    batch.action)
      q = jnp.minimum(q1, q2)
      v = value_network.apply(value_params, batch.observation)
      value_loss = expectile_loss(q - v, expectile).mean()
      return value_loss, {"value_loss": value_loss, "value": v.mean()}

    def critic_loss_fn(
        critic_params: networks_lib.Params,
        target_value_params: networks_lib.Params,
        batch: core_types.Transition,
    ) -> Tuple[jnp.ndarray, Any]:
      next_v = value_network.apply(target_value_params, batch.next_observation)
      target_q = batch.reward + discount * batch.discount * next_v
      q1, q2 = critic_network.apply(critic_params, batch.observation,
                                    batch.action)
      critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
      return critic_loss, {
          "critic_loss": critic_loss,
          "q1": q1.mean(),
          "q2": q2.mean(),
      }

    bc_grad_fn = jax.grad(bc_loss_fn, has_aux=True)
    actor_grad_fn = jax.grad(awr_actor_loss_fn, has_aux=True)
    value_grad_fn = jax.grad(value_loss_fn, has_aux=True)
    critic_grad_fn = jax.grad(critic_loss_fn, has_aux=True)
    
    def bc_update_step(
        state: BCTrainingState,
        batch: core_types.Transition
    ) -> Tuple[BCTrainingState, _Metrics]:
      # Update behavior cloning network
      bc_key, key = jax.random.split(state.key)
      bc_grads, bc_metrics = bc_grad_fn(state.bc_params, bc_key, batch)
      bc_updates, bc_opt_state = policy_optimizer.update(bc_grads, state.bc_opt_state)
      bc_params = optax.apply_updates(state.bc_params, bc_updates)

      state = BCTrainingState(bc_params=bc_params, bc_opt_state=bc_opt_state, key=key)

      return state, bc_metrics

    def update_step(
        state: TrainingState,
        batch: core_types.Transition) -> Tuple[TrainingState, _Metrics]:
      # Update value network first
      policy_key, key = jax.random.split(state.key)
      value_grads, value_metrics = value_grad_fn(state.value_params,
                                                 state.target_critic_params,
                                                 batch)
      value_updates, value_opt_state = value_optimizer.update(
          value_grads, state.value_opt_state)
      value_params = optax.apply_updates(state.value_params, value_updates)
      # Update policy network
      policy_grads, policy_metrics = actor_grad_fn(
          state.policy_params,
          state.old_policy_params,
          policy_key,
          state.target_critic_params,
          value_params,
          batch,
          state.steps
      )
      policy_updates, policy_opt_state = policy_optimizer.update(
          policy_grads, state.policy_opt_state)
      policy_params = optax.apply_updates(state.policy_params, policy_updates)
      # Update critic network
      critic_grads, critic_metrics = critic_grad_fn(state.critic_params,
                                                    value_params, batch)
      critic_updates, critic_opt_state = critic_optimizer.update(
          critic_grads, state.critic_opt_state)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)

      target_critic_params = optax.incremental_update(
          critic_params, state.target_critic_params, tau)
      state = TrainingState(
          policy_params=policy_params,
          policy_opt_state=policy_opt_state,
          old_policy_params=state.old_policy_params,
          old_policy_opt_state=state.old_policy_opt_state,
          critic_params=critic_params,
          critic_opt_state=critic_opt_state,
          value_params=value_params,
          value_opt_state=value_opt_state,
          target_critic_params=target_critic_params,
          key=key,
          steps=state.steps + 1,
      )
      return state, {**critic_metrics, **value_metrics, **policy_metrics}

    self._update_step = jax.jit(update_step)
    self._bc_update_step = jax.jit(bc_update_step)

    def make_initial_state_bc(key):
      bc_key, key = jax.random.split(key)

      bc_params = bc_network.init(bc_key)
      bc_opt_state = policy_optimizer.init(bc_params)

      state = BCTrainingState(
          bc_params=bc_params,
          bc_opt_state=bc_opt_state,
          key=key,
      )
      return state

    def make_initial_state(key, bc_state):
      critic_key, value_key, key = jax.random.split(key, 3)
      old_policy_params = bc_state.bc_params
      old_policy_opt_state = bc_state.bc_opt_state
      policy_params = bc_state.bc_params
      policy_opt_state = bc_state.bc_opt_state
      critic_params = critic_network.init(critic_key)
      critic_opt_state = critic_optimizer.init(critic_params)
      value_params = value_network.init(value_key)
      value_opt_state = value_optimizer.init(value_params)
      state = TrainingState(
          old_policy_params=old_policy_params,
          old_policy_opt_state=old_policy_opt_state,
          policy_params=policy_params,
          policy_opt_state=policy_opt_state,
          critic_params=critic_params,
          critic_opt_state=critic_opt_state,
          target_critic_params=critic_params,
          value_params=value_params,
          value_opt_state=value_opt_state,
          key=key,
          steps=0,
      )
      return state

    self._bc_state = make_initial_state_bc(random_key_bc)
    self._iterator = dataset

    
    for _ in range(self._bc_max_steps):
      transitions = next(self._iterator)
      self._bc_state, metrics = self._bc_update_step(self._bc_state, transitions)

    self._state = make_initial_state(random_key, self._bc_state)
    
    self._logger = logger or loggers.make_default_logger(
        "learner", save_data=False)
    self._counter = counter or counting.Counter()
    self._timestamp = None  


  def update_old_policy(self):
    self._state = self._state._replace(old_policy_params = self._state.policy_params)


  def step(self):
    # Get data from replay
    transitions = next(self._iterator)
    # Perform a single learner step
    self._state, metrics = self._update_step(self._state, transitions)

    # Compute elapsed time
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        "policy": self._state.policy_params,
        "critic": self._state.critic_params,
    }
    return [variables[name] for name in names]

  def restore(self, state: TrainingState):
    self._state = state

  def save(self) -> TrainingState:
    return self._state
