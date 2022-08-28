import ml_collections

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  config.learning_rate = 0.001 # 0.001
  config.momentum = 0.9
  config.batch_size = 100 # For RCO - 128, TAKD - 128, CSKD - 1024
  config.num_epochs = 50 #100
  config.dist_train_epochs = 21
  config.dist_val_epochs = 36
  config.dataset_path = './dataset/'
  config.ckpt_path = f'./checkpoint/'
  return config