from ml_collections import config_dict

_NUM_SEEDS = 10


def get_config():
  config = config_dict.ConfigDict()
  config.batch_size = 64
  config.max_steps = int(1e6)
  config.evaluate_every = int(5e4)
  config.evaluation_episodes = 10
  config.seed =2
  config.use_dataset_reward = False
  config.wandb_project = 'otr'
  config.wandb_entity = None
  #config.expert_dataset_name = 'antmaze-medium-diverse-v0'
  #config.offline_dataset_name = 'antmaze-medium-diverse-v0'
  config.k = 10

  config.squashing_fn = 'exponential'
  config.alpha = 4
  config.beta = 5
  config.normalize_by_atom = False

  # IQL config
  config.opt_decay_schedule = "cosine"
  config.dropout_rate = None
  config.actor_lr = 1.1e-06
  config.value_lr = 1.1e-06
  config.critic_lr = 1.1e-06
  config.hidden_dims = (256, 256)
  config.iql_kwargs = dict(
      discount=0.9,
      expectile=0.7,  # The actual tau for expectiles.
      temperature=6)
  config.log_to_wandb = False
  return config


