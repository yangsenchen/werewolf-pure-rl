from gymnasium.envs.registration import register
from werewolf_env.myenv import MyEnv

register(
    id='WolvesVillagers-v0',
    entry_point='werewolf_env.myenv:MyEnv',
    max_episode_steps=100
)


print("Environment registered.")