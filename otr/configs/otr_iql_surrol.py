from ml_collections import config_dict

_NUM_SEEDS = 10


def get_config():
  config = config_dict.ConfigDict()
  config.batch_size = 256
  config.max_steps = int(1e6)
  config.evaluate_every = int(5e4)
  config.evaluation_episodes = 10
  config.seed = 0
  config.use_dataset_reward = False
  config.wandb_project = 'otr'
  config.wandb_entity = None
  #config.expert_dataset_name = 'antmaze-medium-diverse-v0'
  #config.offline_dataset_name = 'antmaze-medium-diverse-v0'
  config.k = 10

  config.squashing_fn = 'exponential'
  config.alpha = 5.0
  config.beta = 1.0
  config.normalize_by_atom = False

  # IQL config
  config.opt_decay_schedule = "cosine"
  config.dropout_rate = None
  config.actor_lr = 3e-4
  config.value_lr = 3e-4
  config.critic_lr = 3e-4
  config.hidden_dims = (256, 256)
  config.iql_kwargs = dict(
      discount=0.99,
      expectile=0.9,  # The actual tau for expectiles.
      temperature=10.0)
  config.log_to_wandb = False
  return config


