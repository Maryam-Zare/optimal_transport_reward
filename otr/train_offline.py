from absl import app
from absl import flags
import acme
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.jax import variable_utils
from acme.utils import counting
import jax
from ml_collections import config_flags
import numpy as np
import optax
import tqdm

from otr import dataset_utils
from otr import evaluation
from otr import experiment_utils
from otr.agents import iql
from otr.agents.otil import rewarder as rewarder_lib

import pandas as pd

import csv
import os
import optuna
from datetime import datetime
#*********************************************************************************************

_CONFIG = config_flags.DEFINE_config_file("config", "configs/otr_iql_mujoco.py")
_WORKDIR = flags.DEFINE_string('workdir', '/tmp/otr', '')


def relabel_rewards(rewarder, trajectory):
  rewards = rewarder.compute_offline_rewards(trajectory)
  relabeled_transitions = []
  for transition, reward in zip(trajectory, rewards):
      relabeled_transitions.append(transition._replace(reward=reward))
  return relabeled_transitions


def compute_iql_reward_scale(trajs):
  """Rescale rewards based on max/min from the dataset.
  This is also used in the original IQL implementation.
  """
  trajs = trajs.copy()

  def compute_returns(tr):
      return sum([step.reward for step in tr])

  trajs.sort(key=compute_returns)
  reward_scale = 1000.0 / (
      compute_returns(trajs[-1]) - compute_returns(trajs[0]))
  return reward_scale


def get_demonstration_dataset(config):
  """Return the relabeled offline dataset."""

  expert_dataset_name = "/home/ghazaal/Documents/GitHub/SurRoL/surrol/data/demo/data_ActiveTrack-v0_square_100.npz"
  offline_dataset_name = "/home/ghazaal/Documents/GitHub/SurRoL/surrol/data/demo/data_ActiveTrack-v0_square_100.npz"

  if config.use_dataset_reward:
      offline_traj = dataset_utils.convert_dataset_to_trajectories(offline_dataset_name)
      reward_scale = compute_iql_reward_scale(offline_traj)
      reward_bias = 0.0
      relabeled_transitions = dataset_utils.merge_trajectories(offline_traj)
  else:
      offline_traj = dataset_utils.convert_dataset_to_trajectories(expert_dataset_name)
      returns = [sum([t.reward for t in traj]) for traj in offline_traj]
      idx = np.argpartition(returns, -config.k)[-config.k:]
      expert_demo = [offline_traj[i] for i in idx]

      episode_length = 1000
      if config.squashing_fn == 'linear':
          squashing_fn = functools.partial(
              rewarder_lib.squashing_linear, alpha=config.alpha)
      elif config.squashing_fn == 'exponential':
          if config.get("normalize_by_atom", True):
              atom_size = expert_demo[0][0].observation.shape[0]
          else:
              atom_size = 1.0
          squashing_fn = functools.partial(
              rewarder_lib.squashing_exponential,
              alpha=config.alpha,
              beta=config.beta * episode_length / atom_size)
      else:
          raise ValueError(f'Unknown squashing fn {config.squashing_fn}')
      rewarder = rewarder_lib.OTILRewarder(
          expert_demo, episode_length=episode_length, squashing_fn=squashing_fn)

      offline_traj = dataset_utils.convert_dataset_to_trajectories(offline_dataset_name)
      relabeled_trajectories = []
      x = [sum([t.reward for t in traj]) for traj in offline_traj]

      for i in tqdm.trange(len(offline_traj)):
          relabeled_traj = relabel_rewards(rewarder, offline_traj[i])
          relabeled_trajectories.append(relabeled_traj)

      reward_scale = compute_iql_reward_scale(relabeled_trajectories)
      reward_bias = 0.0
      relabeled_transitions = dataset_utils.merge_trajectories(relabeled_trajectories)

  relabeled_transitions = relabeled_transitions._replace(
      reward=relabeled_transitions.reward * reward_scale + reward_bias)
  return relabeled_transitions


def objective_function(trial):
  config = _CONFIG.value
  workdir = _WORKDIR.value
  log_to_wandb = config.log_to_wandb
  

  wandb_kwargs = {
      'project': config.wandb_project,
      'entity': config.wandb_entity,
      'config': config.to_dict(),
  }

  logger_factory = experiment_utils.LoggerFactory(
      workdir=workdir,
      log_to_wandb=log_to_wandb,
      wandb_kwargs=wandb_kwargs,
      learner_time_delta=10,
      evaluator_time_delta=0)

  config.dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
  config.alpha = trial.suggest_uniform('alpha', 1.0, 5.0)
  config.beta = trial.suggest_uniform('beta', 1.0, 5.0)
  config.batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

  dataset = get_demonstration_dataset(config)

  # Create dataset iterator for the relabeled dataset
  key = jax.random.PRNGKey(config.seed)
  key_learner, key_demo, key = jax.random.split(key, 3)

  iterator = dataset_utils.JaxInMemorySampler(dataset, key_demo,
                                              config.batch_size)

  # Create an environment and grab the spec.
  environment = dataset_utils.make_environment(
      'ActiveTrack-v0', seed=config.seed)
  # Create the networks to optimize.
  spec = acme.make_environment_spec(environment)

  networks = iql.make_networks(
      spec, hidden_dims=config.hidden_dims, dropout_rate=config.dropout_rate)

  counter = counting.Counter(time_delta=0.0)

  if config.opt_decay_schedule == "cosine":
      schedule_fn = optax.cosine_decay_schedule(-config.actor_lr,
                                                config.max_steps)
      policy_optimizer = optax.chain(optax.scale_by_adam(),
                                      optax.scale_by_schedule(schedule_fn))
  else:
      policy_optimizer = optax.adam(config.actor_lr)

  global optimization_stage
  config.iql_kwargs = dict(
      temperature=trial.suggest_int('temperature', 0, 15),
      expectile=trial.suggest_uniform('expectile', 0.5, 1.0),
      discount=trial.suggest_uniform('discount', 0.9, 0.99)
  )

  print(f'\n\n\n---------------- Optimization Stage {optimization_stage}-------------------------------------------\n')
  optimization_stage += 1

  # Create the learner.
  learner_counter = counting.Counter(counter, "learner", time_delta=0.0)
  learner = iql.IQLLearner(
      networks=networks,
      random_key=key_learner,
      dataset=iterator,
      policy_optimizer=policy_optimizer,
      critic_optimizer=optax.adam(config.critic_lr),
      value_optimizer=optax.adam(config.value_lr),
      **config.iql_kwargs,
      logger=logger_factory('learner', learner_counter.get_steps_key(), 0),
      counter=learner_counter,
  )

  def evaluator_network(params, key, observation):
      del key
      action_distribution = networks.policy_network.apply(
          params, observation, is_training=False)
      return action_distribution.mode()

  eval_actor = actors.GenericActor(
      actor_core_lib.batched_feed_forward_to_actor_core(evaluator_network),
      random_key=key,
      variable_client=variable_utils.VariableClient(
          learner, "policy", device="cpu"),
      backend="cpu",
  )

  eval_counter = counting.Counter(counter, "eval_loop", time_delta=0.0)
  eval_loop = evaluation.D4RLEvalLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=logger_factory('eval_loop', eval_counter.get_steps_key(), 0),
  )

  # Run the environment loop.
  steps = 0
  while steps < config.max_steps:
      for _ in range(config.evaluate_every):
          learner.step()
      steps += config.evaluate_every
      average_normalized_return = eval_loop.run(config.evaluation_episodes)
  return average_normalized_return


optimization_stage = 1

def save_hyperparameters_to_csv(study, log_dir):
  # Create directory if it doesn't exist
  if not os.path.exists(log_dir):
      os.makedirs(log_dir)
  
  # Get the best trial
  trial = study.best_trial
  
  # Construct a dictionary of the best parameters
  best_params = trial.params
  best_params['value'] = trial.value  # You might also want to store the best objective value

  # Use timestamp to create a unique filename for each set of hyperparameters
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  filename = f"optimal_hyperparameters_{timestamp}.csv"
  filepath = os.path.join(log_dir, filename)
  
  # Write the dictionary to a CSV file
  with open(filepath, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      # Write the header
      writer.writerow(['Parameter', 'Value'])
      # Write each parameter and the final value
      for key, val in best_params.items():
          writer.writerow([key, val])
  
  print(f"Optimal hyperparameters saved to {filepath}")
  
    
def main(argv):
  """Main function to handle flag parsing and initiate the study."""
  # It's crucial to call this to ensure flags are parsed.
  flags.FLAGS(argv)

  # Now proceed with setting up the study and optimizing
  study = optuna.create_study(direction='maximize')
  study.optimize(objective_function, n_trials=10)
  
  save_hyperparameters_to_csv(study, 'optimal_params')

  print("Best trial:")
  trial = study.best_trial
  print("  Value: ", -1 * trial.value)
  print("  Params: ")
  for key, value in trial.params.items():
      print("    {}: {}".format(key, value))

if __name__ == '__main__':
    app.run(main)    

