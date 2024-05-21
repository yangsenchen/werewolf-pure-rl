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
initial_model_path = f'models/{total_players}_werewolf_0'
initial_model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
initial_model.set_logger(new_logger)
initial_model.save(initial_model_path)

# Total timesteps and batch size
total_timesteps = 100000
round_size = 1000
progress_bar_callback = TQDMProgressBarCallback(2047)

# Alternate training between camps
camps_name = ['villager', 'werewolf']
rival_model = PPO.load(initial_model_path)  # 使用随机初始化的模型作为对手
model = PPO.load(initial_model_path) # 使用随机初始化的模型作为训练模型
model.set_logger(new_logger)
current_camp = 0  # 从村民开始

# 开始训练 每个阵营训练round_size轮 然后换下一个阵营 不停循环 左脚踩右脚
for i in tqdm(range(0, total_timesteps, round_size)):
    
    camp_name = camps_name[current_camp]
    print(f"{i} Training {camp_name} camp for {round_size} timesteps...")
    
    # Initialize environment with rival model
    camp = 0 if camp_name == 'villager' else 1
    env = MyEnv(num_wolves=3, num_villagers=6, rival=rival_model, camp=camp, debug_mode=False)
    env = Monitor(env)
    
    
    if i > 0:
        # 使用上一轮训练的模型作为这一轮训练的初始化
        model_path = f'models/{total_players}_{camps_name[current_camp]}_{i-1000}'
        model = PPO.load(model_path)
        model.set_logger(new_logger)
    
    # 训练
    model.set_env(env)
    model.learn(total_timesteps=round_size, callback=[progress_bar_callback])
    
    # 存储模型
    model_path = f'models/{total_players}_{camps_name[camp]}_{i + round_size}'
    model.save(model_path)
    
    # 换下一个阵营训练
    rival_model = PPO.load(model_path)
    current_camp = (current_camp + 1) % 2

print(f"Training completed.")
