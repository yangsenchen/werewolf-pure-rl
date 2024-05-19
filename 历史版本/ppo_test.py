import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from werewolf_env import MyEnv
import numpy as np

# Load the trained models
werewolf_model_path = 'models/9_werewolf_100000'
villager_model_path = 'models/9_villager_99000'
werewolf_model = PPO.load(werewolf_model_path)
villager_model = PPO.load(villager_model_path)

# Number of test games
num_test_games = 100

# Function to test the models
def test_models(werewolf_model, villager_model, num_games=10):
    werewolf_wins = 0
    villager_wins = 0
    draw = 0
    # for i in range(num_games):
    #     print(f"Starting game {i+1}")
    #     # 测试村民 对手为狼人
    #     env = MyEnv(num_wolves=3, num_villagers=6, rival=werewolf_model, camp=0, debug_mode=True)
    #     env = Monitor(env)
    #     obs, _ = env.reset()
    #     done = False
    #     while not done:
    #         action, _ = villager_model.predict(obs, deterministic=True)
    #         obs, rewards, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
        
    #     remaining_wolves = np.sum(env.alive & (env.roles == env.roles_names.index('Wolf')))
    #     remaining_villagers = np.sum(env.alive & (env.roles == env.roles_names.index('Villager')))
    #     if remaining_wolves > 0:
    #         werewolf_wins += 1
    #         print("Werewolves win!")
    #     else:
    #         villager_wins += 1
    #         print("Villagers win!")

    # print(f"Werewolf win rate: {werewolf_wins / num_games:.2f}")
    # print(f"Villager win rate: {villager_wins / num_games:.2f}")

    for i in range(num_games):
        print(f"Starting game {i+1}")
        # 测试狼人 对手为村民
        env = MyEnv(num_wolves=3, num_villagers=6, rival=villager_model, camp=1, debug_mode=True)
        env = Monitor(env)
        obs, _ = env.reset()
        done = False
        cnt = 0
        while not done:
            cnt += 1
            action, _ = werewolf_model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if cnt > 100:
                draw += 1
                break
        
        remaining_wolves = np.sum(env.alive & (env.roles == env.roles_names.index('狼人')))
        remaining_villagers = np.sum(env.alive & (env.roles == env.roles_names.index('村民')))
        if remaining_villagers == 0:
            werewolf_wins += 1
            print("Werewolves win!")
        elif remaining_wolves == 0:
            villager_wins += 1
            print("Villagers win!")

    print(f"Draw: {draw}")
    print(f"Werewolf win rate: {werewolf_wins / num_games:.2f}")
    print(f"Villager win rate: {villager_wins / num_games:.2f}")

# Test the models
test_models(werewolf_model, villager_model, num_test_games)
