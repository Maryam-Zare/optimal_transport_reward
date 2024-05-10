from ml_collections import config_dict

_NUM_SEEDS = 10


def get_config():
  config = config_dict.ConfigDict()
  config.batch_size = 256
  config.max_steps = int(1e6)
  config.evaluate_every = int(5e4)
  config.evaluation_episodes = 10
  config.seed = 4
  config.use_dataset_reward = False
  config.wandb_project = 'otr'
  config.wandb_entity = None
  #config.expert_dataset_name = 'antmaze-medium-diverse-v0'
  #config.offline_dataset_name = 'antmaze-medium-diverse-v0'
  config.k = 10

  config.squashing_fn = 'exponential'
  config.alpha = 2.1
  config.beta = 1.98
  config.normalize_by_atom = False

  # IQL config
  config.opt_decay_schedule = "cosine"
  config.dropout_rate = 0.05
  config.actor_lr = 3e-4
  config.value_lr = 3e-4
  config.critic_lr = 3e-4
  config.hidden_dims = (256, 256)
  config.iql_kwargs = dict(
      discount=0.94,
      expectile=0.52,  # The actual tau for expectiles.
      temperature=491)
  config.log_to_wandb = False
  return config


