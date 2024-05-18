# # # ppo_wolves_villagers.py
# # import gymnasium as gym
# # from stable_baselines3 import PPO
# # from stable_baselines3.common.callbacks import CheckpointCallback
# # from stable_baselines3.common.monitor import Monitor
# # import myenv  # Ensure the environment is imported to register it

# # # Set CUDA debugging flag
# # import os
# # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# # # Create the Wolves-Villagers environment
# # env_id = 'WolvesVillagers-v0'
# # env = gym.make(env_id)
# # env = Monitor(env)

# # # Create PPO model with MultiInputPolicy
# # model = PPO("MultiInputPolicy", env, verbose=1)

# # # Create checkpoint callback
# # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_wolves_villagers')

# # # Train the model
# # model.learn(total_timesteps=100000, callback=checkpoint_callback)

# # # Save the final model
# # model.save('ppo_wolves_villagers_final')

# # # Test the trained model
# # obs, _ = env.reset()

# # for _ in range(100):
# #     action, _states = model.predict(obs, deterministic=True)
# #     obs, rewards, terminated, truncated, info = env.step(action)
# #     if terminated or truncated:
# #         obs, _ = env.reset()

# # env.close()


# # ppo_wolves_villagers.py

# # ppo_wolves_villagers.py
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.logger import configure
# import wandb
# from wandb.integration.sb3 import WandbCallback
# import myenv  # Ensure the environment is imported to register it

# # Initialize Wandb
# wandb.init(
#     project="werewolf-training",
#     entity="yangsenchen",  # Replace this with your Wandb username
#     config={
#         "env_id": "WolvesVillagers-v0",
#         "total_timesteps": 100000,
#         "policy": "MultiInputPolicy",
#         "algorithm": "PPO"
#     }
# )

# # Set CUDA debugging flag
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# # Create the Wolves-Villagers environment
# env_id = 'WolvesVillagers-v0'
# env = gym.make(env_id)
# env = Monitor(env)

# # Configure the logging directory and logger
# log_dir = "./ppo_logs/"
# new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# # Create PPO model with MultiInputPolicy
# model = PPO("MultiInputPolicy", env, verbose=1)
# model.set_logger(new_logger)

# # Create checkpoint callback
# checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_wolves_villagers')

# # Wandb callback
# wandb_callback = WandbCallback(
#     gradient_save_freq=1000,
#     model_save_path='./models_2/',
#     verbose=2
# )

# # Train the model
# total_timesteps = 10000
# print(f"Starting training for {total_timesteps} timesteps...")
# model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])
# print(f"Training completed.")

# # Save the final model
# model_path = 'ppo_wolves_villagers_final'
# model.save(model_path)
# print(f"Model saved at {model_path}.")

# # Test the trained model
# obs, _ = env.reset()
# print(f"Testing the trained model...")

# for _ in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, _ = env.reset()

# env.close()
# print("Testing completed.")



# ppo_wolves_villagers.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import wandb
from wandb.integration.sb3 import WandbCallback
import myenv  # Ensure the environment is imported to register it

# Initialize Wandb
wandb.init(
    project="werewolf-training",
    entity="yangsenchen",
    config={
        "env_id": "WolvesVillagers-v0",
        "total_timesteps": 100000,
        "policy": "MultiInputPolicy",
        "algorithm": "PPO"
    }
)

# Create the Wolves-Villagers environment
env_id = 'WolvesVillagers-v0'
env = gym.make(env_id)
env = Monitor(env)

# Configure the logging directory and logger
log_dir = "./ppo_logs/"
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# Create PPO model with MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1)
model.set_logger(new_logger)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_wolves_villagers')

# Wandb callback
wandb_callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_path='./models/',
    verbose=2
)

# Train the model
total_timesteps = 100000
print(f"Starting training for {total_timesteps} timesteps...")
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])
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
