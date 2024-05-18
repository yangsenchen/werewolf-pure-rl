from gymnasium.envs.registration import register

register(
    id='WolvesVillagers-v0',
    entry_point='werewolf_env.myenv:MyEnv',
    max_episode_steps=100
)


print("Environment registered.")