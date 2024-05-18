

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import werewolf_env  # Ensure the environment is imported to register it


env_id = 'WolvesVillagers-v0'
env = gym.make(env_id)
env = Monitor(env)

# 这个是stable baselines3的logger
log_dir = "./logs/"
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# 创建 PPO model with MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1)
model.set_logger(new_logger)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_wolves_villagers')

# 训练
total_timesteps = 100000
print(f"Starting training for {total_timesteps} timesteps...")
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback]) #, wandb_callback
print(f"Training completed.")

# Save the final model
model_path = 'ppo_wolves_villagers_final'
model.save(model_path)
print(f"Model saved at {model_path}.")

# Test the trained model
obs, _ = env.reset()
print(f"Testing the trained model...")

for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
print("Testing completed.")
