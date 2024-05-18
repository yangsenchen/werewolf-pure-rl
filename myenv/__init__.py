# myenv/__init__.py
from gymnasium.envs.registration import register

register(
    id='WolvesVillagers-v0',
    entry_point='myenv.myenv:MyEnv',
    max_episode_steps=100
)
