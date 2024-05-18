import gymnasium as gym
from stable_baselines3 import PPO
import werewolf_env  # Ensure the environment is imported to register it

# Load the trained model
model_path = 'ppo_wolves_villagers_final'
model = PPO.load(model_path)

# Initialize the environment
env_id = 'WolvesVillagers-v0'
env = gym.make(env_id)

# Function to simulate a single game
def simulate_game(env, model):
    obs, _ = env.reset()
    done = False
    truncated = False
    rewards = []
    step_count = 0

    while not done and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        step_count += 1

    return rewards, step_count

# Simulate 5 games and check the behavior
num_games = 5
results = []

print(f"Simulating {num_games} games...")

for i in range(num_games):
    rewards, steps = simulate_game(env, model)
    results.append((rewards, steps))
    print(f"Game {i + 1}: Rewards = {rewards}, Steps = {steps}")

env.close()

# Print summary of results
print("\nSummary of results:")
for i, (rewards, steps) in enumerate(results):
    print(f"Game {i + 1}:")
    print(f"  Rewards = {rewards}")
    print(f"  Steps = {steps}")

print("Simulation completed.")
