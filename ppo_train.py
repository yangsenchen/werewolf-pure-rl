import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from werewolf_env import MyEnv  # Ensure the environment is imported to register it
from tqdm import tqdm
from where_gpu import get_free_gpu
import os
import numpy as np

# Setup GPU
free_gpu = get_free_gpu()
if free_gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(free_gpu)
else:
    print("No GPU available or error occurred.")

# Function to evaluate the model and calculate win rate
def evaluate_model(model, env_id, rival_model, camp, num_episodes=100):
    env = MyEnv(num_wolves=3, num_villagers=6, rival=rival_model, camp=0 if camp == 'werewolf' else 1, debug_mode=True)
    env = Monitor(env)
    wins = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        cnt = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cnt += 1
            if cnt > 20:
                break
        if camp == 'werewolf' and np.sum(env.alive & (env.roles == env.roles_names.index('Wolf'))) > 0:
            wins += 1
        elif camp == 'villager' and np.sum(env.alive & (env.roles == env.roles_names.index('Wolf'))) == 0:
            wins += 1
    win_rate = wins / num_episodes
    print(f"{camp.capitalize()} win rate: {win_rate:.2f}")
    return win_rate

# Progress bar callback
class TQDMProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super(TQDMProgressBarCallback, self).__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

# Initialize environment and logger
env_id = 'WolvesVillagers-v0'
env = gym.make(env_id)
env = Monitor(env)
log_dir = "./logs/"
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

total_players = 9
# Create and save initial model
initial_model_path = f'models/{total_players}_villager_0'
initial_model = PPO("MultiInputPolicy", env, verbose=1)
initial_model.save(initial_model_path)

# Total timesteps and batch size
total_timesteps = 100000
batch_size = 1000
progress_bar_callback = TQDMProgressBarCallback(2047)

# Alternate training between camps
camps = ['werewolf', 'villager']
current_camp = 0  # Start with werewolf camp
rival_model = PPO.load(initial_model_path)  # Use initial model as the first rival
model = PPO.load(initial_model_path)  # Load initial model for the first iteration

print(f"Starting training for {total_timesteps} timesteps...")

for i in tqdm(range(0, total_timesteps, batch_size)):
    # Update camp
    camp = camps[current_camp]
    print(f"{i} Training {camp} camp for {batch_size} timesteps...")
    current_camp = (current_camp + 1) % 2
    
    # Initialize environment with rival model
    env = MyEnv(num_wolves=3, num_villagers=6, rival=rival_model, camp=0 if camp == 'werewolf' else 1, debug_mode=False)
    env = Monitor(env)
    
    # Load the model from the previous iteration
    if i > 0:
        model_path = f'models/{total_players}_{camps[(current_camp + 1) % 2]}_{i-1000}'
        model = PPO.load(model_path)
    
    # Train the model
    model.set_env(env)
    model.set_logger(new_logger)
    model.learn(total_timesteps=batch_size, callback=[progress_bar_callback])
    
    # Save the model and update rival model
    model_path = f'models/{total_players}_{camp}_{i + batch_size}'
    model.save(model_path)
    rival_model = PPO.load(model_path)

    # Print info during training
    # obs, _ = env.reset()
    # done = False
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, rewards, terminated, truncated, info = env.step(action)
    #     print(f"Info: {info}")
    #     done = terminated or truncated

    # Evaluate the model and calculate the win rate
    # win_rate = evaluate_model(model, env_id, rival_model, camp)
    # print(f"Win rate after {i + batch_size} timesteps: {win_rate:.2f}")

print(f"Training completed.")

# Save the final model
final_model_path = f'models/{total_players}_wolves_villagers_final'
model.save(final_model_path)
print(f"Model saved at {final_model_path}.")

# Test the trained model
# env = gym.make(env_id)
# env = Monitor(env)
# obs, _ = env.reset()
# print(f"Testing the trained model...")

# for _ in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, _ = env.reset()

# env.close()
# print("Testing completed.")
