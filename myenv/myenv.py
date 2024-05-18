# # # # # # myenv/myenv.py
# # # # # import gymnasium as gym
# # # # # from gymnasium import spaces
# # # # # import numpy as np

# # # # # class MyEnv(gym.Env):
# # # # #     def __init__(self, num_wolves=2, num_villagers=4):
# # # # #         super(MyEnv, self).__init__()

# # # # #         # Game Parameters
# # # # #         self.num_wolves = num_wolves
# # # # #         self.num_villagers = num_villagers
# # # # #         self.num_players = self.num_wolves + self.num_villagers

# # # # #         # Roles
# # # # #         self.roles = ['Villager', 'Wolf']
# # # # #         self.role_distribution = {'Villager': self.num_villagers, 'Wolf': self.num_wolves}

# # # # #         # Observation Space
# # # # #         self.observation_space = spaces.Dict({
# # # # #             'players': spaces.MultiDiscrete([len(self.roles)] * self.num_players),
# # # # #             'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
# # # # #             'game_phase': spaces.Discrete(2)  # 0: Day, 1: Night
# # # # #         })

# # # # #         # Action Space
# # # # #         self.action_space = spaces.Discrete(self.num_players)  # Target player index

# # # # #         # Game State
# # # # #         self.players = np.zeros(self.num_players, dtype=int)  # Store role indices for each player
# # # # #         self.votes = np.zeros(self.num_players, dtype=int)
# # # # #         self.game_phase = 0  # 0: Day, 1: Night
# # # # #         self.current_player = 0

# # # # #         self.reset()

# # # # #     def reset(self, seed=None, options=None):
# # # # #         """Reset the game to the initial state."""
# # # # #         super().reset(seed=seed)
# # # # #         # Initialize roles
# # # # #         self.players = self._assign_roles()
# # # # #         self.votes = np.zeros(self.num_players, dtype=int)
# # # # #         self.game_phase = 0  # Start with day phase
# # # # #         self.current_player = 0

# # # # #         return self._get_observation(), {}

# # # # #     def _assign_roles(self):
# # # # #         """Assign roles to players randomly based on role distribution."""
# # # # #         roles = []
# # # # #         for role, count in self.role_distribution.items():
# # # # #             roles.extend([self.roles.index(role)] * count)
# # # # #         roles = np.random.permutation(roles)
# # # # #         return roles

# # # # #     def _get_observation(self):
# # # # #         """Get the current observation."""
# # # # #         return {
# # # # #             'players': np.clip(self.players, 0, len(self.roles) - 1),
# # # # #             'votes': np.clip(self.votes, 0, self.num_players - 1),
# # # # #             'game_phase': np.clip(self.game_phase, 0, 1)
# # # # #         }

# # # # #     def step(self, action):
# # # # #         """Take an action and return the observation, reward, terminated, truncated, and info."""
# # # # #         if action < 0 or action >= self.num_players:
# # # # #             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}")

# # # # #         terminated = False
# # # # #         truncated = False
# # # # #         info = {}

# # # # #         if self.game_phase == 0:  # Day phase
# # # # #             # Vote to eliminate a player
# # # # #             self.votes[action] += 1

# # # # #             # Check if all players have voted
# # # # #             if self.current_player == self.num_players - 1:
# # # # #                 eliminated_player = np.argmax(self.votes)
# # # # #                 reward, terminated = self._eliminate_player(eliminated_player)

# # # # #                 # Move to night phase
# # # # #                 self.game_phase = 1
# # # # #                 self.current_player = 0
# # # # #                 self.votes = np.zeros(self.num_players, dtype=int)
# # # # #             else:
# # # # #                 reward = 0
# # # # #                 terminated = False
# # # # #                 self.current_player += 1
# # # # #         else:  # Night phase
# # # # #             reward, terminated = self._handle_night_phase(action)
# # # # #             self.game_phase = 0
# # # # #             self.current_player = 0

# # # # #         return self._get_observation(), reward, terminated, truncated, info

# # # # #     def _eliminate_player(self, player):
# # # # #         """Eliminate the player and return the reward and terminated status."""
# # # # #         role = self.players[player]
# # # # #         self.players[player] = -1  # Mark as eliminated

# # # # #         if role == self.roles.index('Wolf'):
# # # # #             reward = 1
# # # # #         else:
# # # # #             reward = -1

# # # # #         # Check win conditions
# # # # #         remaining_wolves = np.sum(self.players == self.roles.index('Wolf'))
# # # # #         remaining_villagers = np.sum(self.players == self.roles.index('Villager'))

# # # # #         if remaining_wolves == 0:
# # # # #             return reward, True  # Villagers win
# # # # #         elif remaining_wolves >= remaining_villagers:
# # # # #             return reward, True  # Wolves win
# # # # #         else:
# # # # #             return reward, False  # Game continues

# # # # #     def _handle_night_phase(self, action):
# # # # #         """Handle wolves attacking during the night phase."""
# # # # #         wolves_indices = np.where(self.players == self.roles.index('Wolf'))[0]

# # # # #         if action < 0 or action >= self.num_players:
# # # # #             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}")

# # # # #         if self.current_player in wolves_indices:
# # # # #             self.players[action] = -1  # Wolves attack a target
# # # # #             reward = 1
# # # # #         else:
# # # # #             reward = 0

# # # # #         terminated = False
# # # # #         remaining_wolves = np.sum(self.players == self.roles.index('Wolf'))
# # # # #         remaining_villagers = np.sum(self.players == self.roles.index('Villager'))

# # # # #         if remaining_wolves == 0 or remaining_wolves >= remaining_villagers:
# # # # #             terminated = True

# # # # #         return reward, terminated



# # # # # myenv/myenv.py
# # # # import gymnasium as gym
# # # # from gymnasium import spaces
# # # # import numpy as np

# # # # class MyEnv(gym.Env):
# # # #     def __init__(self, num_wolves=2, num_villagers=4):
# # # #         super(MyEnv, self).__init__()

# # # #         # Game Parameters
# # # #         self.num_wolves = num_wolves
# # # #         self.num_villagers = num_villagers
# # # #         self.num_players = self.num_wolves + self.num_villagers

# # # #         # Roles
# # # #         self.roles = ['Villager', 'Wolf']
# # # #         self.role_distribution = {'Villager': self.num_villagers, 'Wolf': self.num_wolves}

# # # #         # Observation Space
# # # #         self.observation_space = spaces.Dict({
# # # #             'players': spaces.MultiDiscrete([len(self.roles)] * self.num_players),
# # # #             'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
# # # #             'game_phase': spaces.Discrete(2)  # 0: Day, 1: Night
# # # #         })

# # # #         # Action Space
# # # #         self.action_space = spaces.Discrete(self.num_players)  # Target player index

# # # #         # Game State
# # # #         self.players = np.zeros(self.num_players, dtype=int)  # Store role indices for each player
# # # #         self.votes = np.zeros(self.num_players, dtype=int)
# # # #         self.game_phase = 0  # 0: Day, 1: Night
# # # #         self.current_player = 0

# # # #         # Logging
# # # #         self.episode_log = []
# # # #         self.day = 1

# # # #         self.reset()

# # # #     def reset(self, seed=None, options=None):
# # # #         """Reset the game to the initial state."""
# # # #         super().reset(seed=seed)
# # # #         # Initialize roles
# # # #         self.players = self._assign_roles()
# # # #         self.votes = np.zeros(self.num_players, dtype=int)
# # # #         self.game_phase = 0  # Start with day phase
# # # #         self.current_player = 0
# # # #         self.day = 1
# # # #         self.episode_log = [f"Day {self.day}:"]

# # # #         return self._get_observation(), {}

# # # #     def _assign_roles(self):
# # # #         """Assign roles to players randomly based on role distribution."""
# # # #         roles = []
# # # #         for role, count in self.role_distribution.items():
# # # #             roles.extend([self.roles.index(role)] * count)
# # # #         roles = np.random.permutation(roles)
# # # #         return roles

# # # #     def _get_observation(self):
# # # #         """Get the current observation."""
# # # #         return {
# # # #             'players': np.clip(self.players, 0, len(self.roles) - 1),
# # # #             'votes': np.clip(self.votes, 0, self.num_players - 1),
# # # #             'game_phase': np.clip(self.game_phase, 0, 1)
# # # #         }

# # # #     def step(self, action):
# # # #         """Take an action and return the observation, reward, terminated, truncated, and info."""
# # # #         if action < 0 or action >= self.num_players:
# # # #             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}")

# # # #         terminated = False
# # # #         truncated = False
# # # #         info = {}

# # # #         if self.game_phase == 0:  # Day phase
# # # #             # Vote to eliminate a player
# # # #             self.votes[action] += 1
# # # #             player_role = self.roles[self.players[self.current_player]]
# # # #             log = f"    Player {self.current_player} ({player_role}): votes to exile Player {action}"
# # # #             self.episode_log.append(log)

# # # #             # Check if all players have voted
# # # #             if self.current_player == self.num_players - 1:
# # # #                 eliminated_player = np.argmax(self.votes)
# # # #                 reward, terminated = self._eliminate_player(eliminated_player)

# # # #                 # Move to night phase
# # # #                 self.game_phase = 1
# # # #                 self.current_player = 0
# # # #                 self.votes = np.zeros(self.num_players, dtype=int)
# # # #                 self.episode_log.append("Night:")
# # # #             else:
# # # #                 reward = 0
# # # #                 terminated = False
# # # #                 self.current_player += 1
# # # #         else:  # Night phase
# # # #             reward, terminated = self._handle_night_phase(action)
# # # #             self.game_phase = 0
# # # #             self.current_player = 0
# # # #             self.day += 1
# # # #             self.episode_log.append(f"Day {self.day}:")

# # # #         return self._get_observation(), reward, terminated, truncated, info

# # # #     def _eliminate_player(self, player):
# # # #         """Eliminate the player and return the reward and terminated status."""
# # # #         role = self.roles[self.players[player]]
# # # #         log = f"    Player {player} ({role}): eliminated"
# # # #         self.episode_log.append(log)

# # # #         self.players[player] = -1  # Mark as eliminated

# # # #         if role == 'Wolf':
# # # #             reward = 1
# # # #         else:
# # # #             reward = -1

# # # #         # Check win conditions
# # # #         remaining_wolves = np.sum(self.players == self.roles.index('Wolf'))
# # # #         remaining_villagers = np.sum(self.players == self.roles.index('Villager'))

# # # #         if remaining_wolves == 0:
# # # #             self.episode_log.append("Villagers win!")
# # # #             return reward, True
# # # #         elif remaining_wolves >= remaining_villagers:
# # # #             self.episode_log.append("Wolves win!")
# # # #             return reward, True
# # # #         else:
# # # #             return reward, False

# # # #     def _handle_night_phase(self, action):
# # # #         """Handle wolves attacking during the night phase."""
# # # #         wolves_indices = np.where(self.players == self.roles.index('Wolf'))[0]

# # # #         if action < 0 or action >= self.num_players:
# # # #             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}")

# # # #         if self.current_player in wolves_indices:
# # # #             self.players[action] = -1  # Wolves attack a target
# # # #             log = f"    Player {self.current_player} (Wolf): votes to attack Player {action}"
# # # #             self.episode_log.append(log)
# # # #             reward = 1
# # # #         else:
# # # #             reward = 0

# # # #         terminated = False
# # # #         remaining_wolves = np.sum(self.players == self.roles.index('Wolf'))
# # # #         remaining_villagers = np.sum(self.players == self.roles.index('Villager'))

# # # #         if remaining_wolves == 0 or remaining_wolves >= remaining_villagers:
# # # #             terminated = True

# # # #         return reward, terminated


# # # # myenv/myenv.py


# # # # myenv/myenv.py
# # # import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np

# # # # myenv/myenv.py
# # # import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np

# # # class MyEnv(gym.Env):
# # #     def __init__(self, num_wolves=2, num_villagers=4):
# # #         super(MyEnv, self).__init__()

# # #         # Game Parameters
# # #         self.num_wolves = num_wolves
# # #         self.num_villagers = num_villagers
# # #         self.num_players = self.num_wolves + self.num_villagers

# # #         # Roles
# # #         self.roles = ['Villager', 'Wolf']
# # #         self.role_distribution = {'Villager': self.num_villagers, 'Wolf': self.num_wolves}

# # #         # Observation Space
# # #         self.observation_space = spaces.Dict({
# # #             'players': spaces.MultiDiscrete([len(self.roles) + 1] * self.num_players),  # +1 for "unknown"
# # #             'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
# # #             'game_phase': spaces.Discrete(2)  # 0: Day, 1: Night
# # #         })

# # #         # Action Space
# # #         self.action_space = spaces.Discrete(self.num_players)  # Target player index

# # #         # Game State
# # #         self.players = np.zeros(self.num_players, dtype=int)  # Store role indices for each player
# # #         self.votes = np.zeros(self.num_players, dtype=int)
# # #         self.game_phase = 0  # 0: Day, 1: Night
# # #         self.current_player = 0

# # #         # Logging
# # #         self.episode_log = []
# # #         self.day = 1

# # #         self.reset()

# # #     def reset(self, seed=None, options=None):
# # #         """Reset the game to the initial state."""
# # #         super().reset(seed=seed)
# # #         # Initialize roles
# # #         self.players = self._assign_roles()
# # #         self.votes = np.zeros(self.num_players, dtype=int)
# # #         self.game_phase = 1  # Start with night phase
# # #         self.current_player = 0
# # #         self.day = 1
# # #         self.episode_log = [f"Day {self.day}:"]
        
# # #         return self._get_observation(), {}

# # #     def _assign_roles(self):
# # #         """Assign roles to players randomly based on role distribution."""
# # #         roles = []
# # #         for role, count in self.role_distribution.items():
# # #             roles.extend([self.roles.index(role)] * count)
# # #         roles = np.random.permutation(roles)
# # #         return roles

# # #     def _get_observation(self):
# # #         """Get the current observation based on the current player's role."""
# # #         current_role = self.players[self.current_player]
# # #         if current_role == self.roles.index('Wolf'):
# # #             # Wolves see everyone's role
# # #             player_roles = np.clip(self.players, 0, len(self.roles) - 1)
# # #         else:
# # #             # Villagers see everyone as "unknown" (+1 for "unknown")
# # #             player_roles = np.full(self.num_players, len(self.roles))

# # #         observation = {
# # #             'players': player_roles,
# # #             'votes': np.clip(self.votes, 0, self.num_players - 1),
# # #             'game_phase': np.clip(self.game_phase, 0, 1)
# # #         }
# # #         # Print observation for debugging purposes
# # #         # print(f"Observation (player {self.current_player}): {observation}")
# # #         return observation

# # #     def step(self, action):
# # #         """Take an action and return the observation, reward, terminated, truncated, and info."""
# # #         if action < 0 or action >= self.num_players:
# # #             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}")

# # #         terminated = False
# # #         truncated = False
# # #         info = {}

# # #         if self.game_phase == 0:  # Day phase
# # #             # Vote to eliminate a player
# # #             self.votes[action] += 1
# # #             player_role = self.roles[self.players[self.current_player]]
# # #             log = f"    Player {self.current_player} ({player_role}): votes to exile Player {action}"
# # #             self.episode_log.append(log)

# # #             # Check if all players have voted
# # #             if self.current_player == self.num_players - 1:
# # #                 eliminated_player = np.argmax(self.votes)
# # #                 reward, terminated = self._eliminate_player(eliminated_player)

# # #                 # Move to night phase
# # #                 self.game_phase = 1
# # #                 self.current_player = 0
# # #                 self.votes = np.zeros(self.num_players, dtype=int)
# # #                 self.episode_log.append("Night:")
# # #             else:
# # #                 reward = 0
# # #                 terminated = False
# # #                 self.current_player += 1
# # #         else:  # Night phase
# # #             reward, terminated = self._handle_night_phase(action)
# # #             self.game_phase = 0
# # #             self.current_player = 0
# # #             self.day += 1
# # #             self.episode_log.append(f"Day {self.day}:")

# # #         return self._get_observation(), reward, terminated, truncated, info

# # #     def _eliminate_player(self, player):
# # #         """Eliminate the player and return the reward and terminated status."""
# # #         role = self.roles[self.players[player]]
# # #         log = f"    Player {player} ({role}): eliminated"
# # #         self.episode_log.append(log)

# # #         self.players[player] = -1  # Mark as eliminated

# # #         wolves_indices = np.where(self.players == self.roles.index('Wolf'))[0]
# # #         if self.current_player in wolves_indices:
# # #             if role == 'Wolf':
# # #                 reward = -1
# # #             else:
# # #                 reward = 1
# # #         else:
# # #             if role == 'Wolf':
# # #                 reward = 1
# # #             else:
# # #                 reward = -1

# # #         # Check win conditions
# # #         remaining_wolves = np.sum(self.players == self.roles.index('Wolf'))
# # #         remaining_villagers = np.sum(self.players == self.roles.index('Villager'))

# # #         if remaining_wolves == 0:
# # #             self.episode_log.append("Villagers win!")
# # #             return reward, True
# # #         elif remaining_wolves >= remaining_villagers:
# # #             self.episode_log.append("Wolves win!")
# # #             return reward, True
# # #         else:
# # #             return reward, False

# # #     def _handle_night_phase(self, action):
# # #         """Handle wolves attacking during the night phase."""
# # #         wolves_indices = np.where(self.players == self.roles.index('Wolf'))[0]

# # #         if action < 0 or action >= self.num_players:
# # #             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}")

# # #         if self.current_player in wolves_indices:
# # #             reward = 0

# # #         else:
# # #             self.players[action] = -1  # Wolves attack a target
# # #             log = f"    Player {self.current_player} (Wolf): votes to attack Player {action}"
# # #             self.episode_log.append(log)
# # #             reward = 1

# # #         terminated = False
# # #         remaining_wolves = np.sum(self.players == self.roles.index('Wolf'))
# # #         remaining_villagers = np.sum(self.players == self.roles.index('Villager'))

# # #         if remaining_wolves == 0 or remaining_wolves >= remaining_villagers:
# # #             terminated = True

# # #         return reward, terminated






# # # myenv/myenv.py
# # import gymnasium as gym
# # from gymnasium import spaces
# # import numpy as np

# # class MyEnv(gym.Env):
# #     def __init__(self, num_wolves=2, num_villagers=4):
# #         super(MyEnv, self).__init__()

# #         # Game Parameters
# #         self.num_wolves = num_wolves
# #         self.num_villagers = num_villagers
# #         self.num_players = self.num_wolves + self.num_villagers

# #         # Roles
# #         self.roles = ['Villager', 'Wolf']
# #         self.role_distribution = {'Villager': self.num_villagers, 'Wolf': self.num_wolves}

# #         # Observation Space
# #         self.observation_space = spaces.Dict({
# #             'players': spaces.MultiDiscrete([len(self.roles) + 1] * self.num_players),  # +1 for "unknown"
# #             'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
# #             'game_phase': spaces.Discrete(2)  # 0: Day, 1: Night
# #         })

# #         # Action Space
# #         self.action_space = spaces.Discrete(self.num_players)  # Target player index

# #         # Game State
# #         self.players = np.zeros(self.num_players, dtype=int)  # Store role indices for each player
# #         self.votes = np.zeros(self.num_players, dtype=int)
# #         self.game_phase = 1  # Start with night phase
# #         self.current_player = 0
# #         self.alive = np.ones(self.num_players, dtype=bool)

# #         # Logging
# #         self.episode_log = []
# #         self.day = 1

# #         self.reset()

# #     def reset(self, seed=None, options=None):
# #         """Reset the game to the initial state."""
# #         super().reset(seed=seed)
# #         # Initialize roles
# #         self.players = self._assign_roles()
# #         self.votes = np.zeros(self.num_players, dtype=int)
# #         self.game_phase = 1  # Start with night phase
# #         self.current_player = 0
# #         self.alive = np.ones(self.num_players, dtype=bool)
# #         self.day = 1
# #         # self.episode_log = [f"Night {self.day}:"]

# #         return self._get_observation(), {}

# #     def _assign_roles(self):
# #         """Assign roles to players randomly based on role distribution."""
# #         roles = []
# #         for role, count in self.role_distribution.items():
# #             roles.extend([self.roles.index(role)] * count)
# #         roles = np.random.permutation(roles)
# #         return roles

# #     def _get_observation(self):
# #         """Get the current observation based on the current player's role."""
# #         current_role = self.players[self.current_player]
# #         if current_role == self.roles.index('Wolf'):
# #             # Wolves see everyone's role
# #             player_roles = np.where(self.alive, self.players, len(self.roles))
# #         else:
# #             # Villagers see everyone as "unknown" (+1 for "unknown")
# #             player_roles = np.where(self.alive, len(self.roles), len(self.roles))

# #         observation = {
# #             'players': player_roles,
# #             'votes': np.clip(self.votes, 0, self.num_players - 1),
# #             'game_phase': np.clip(self.game_phase, 0, 1)
# #         }
# #         return observation

# #     def step(self, action):
# #         """Take an action and return the observation, reward, terminated, truncated, and info."""
# #         if action < 0 or action >= self.num_players: # 
# #             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}.")

# #         terminated = False
# #         truncated = False
# #         info = {}

# #         if self.game_phase == 0:  # Day phase
# #             self.episode_log.append("Day:")

# #             # Vote to eliminate a player
# #             self.votes[action] += 1
# #             player_role = self.roles[self.players[self.current_player]]
# #             log = f"    Player {self.current_player} ({player_role}): votes to exile Player {action}"
# #             self.episode_log.append(log)

# #             # Check if all players have voted
# #             if self.current_player == self.num_players - 1:
# #                 alive_voters = np.where(self.alive)[0]
# #                 eliminated_player = alive_voters[np.argmax(self.votes[alive_voters])]

# #                 role = self.roles[self.players[eliminated_player]]
# #                 log = f"    Player {eliminated_player} ({role}): eliminated"
# #                 self.episode_log.append(log)

# #                 self.alive[eliminated_player] = False
# #                 wolves_indices = np.where((self.players == self.roles.index('Wolf')) & self.alive)[0]

# #                 if self.current_player not in wolves_indices:

# #                     if role == 'Wolf':
# #                         reward = 1
# #                     else:
# #                         reward = -1
# #                 else:
# #                     if role == 'Wolf':
# #                         reward = 1
# #                     else:
# #                         reward = -1
                    
# #                 # Check win conditions
# #                 remaining_wolves = np.sum(self.alive & (self.players == self.roles.index('Wolf')))
# #                 remaining_villagers = np.sum(self.alive & (self.players == self.roles.index('Villager')))

# #                 if remaining_wolves == 0:
# #                     self.episode_log.append("Villagers win!")
# #                     terminated = True
# #                 elif remaining_villagers == 0:
# #                     self.episode_log.append("Wolves win!")
# #                     terminated = True
# #                 else:
# #                     terminated = False

# #                 # Move to night phase
# #                 self.game_phase = 1
# #                 self.current_player = 0
# #                 self.votes = np.zeros(self.num_players, dtype=int)
# #             else:
# #                 reward = 0
# #                 terminated = False
# #                 self.current_player += 1
# #         else:  # Night phase
# #             self.episode_log.append("Night:")
# #             wolves_indices = np.where((self.players == self.roles.index('Wolf')) & self.alive)[0]
# #             current_role = self.players[self.current_player]

# #             if self.current_player in wolves_indices:
# #                 self.night_votes[action] += 1
# #                 log = f"    Player {self.current_player} (Wolf): votes to attack Player {action}"
# #                 self.episode_log.append(log)
# #                 reward = 0
# #             else:
# #                 # Villagers do nothing at night
# #                 log = f"    Player {self.current_player} ({self.roles[current_role]}): does nothing at night"
# #                 self.episode_log.append(log)
# #                 reward = 1

# #             if self.current_player == self.num_players - 1:
# #                 # Eliminate the player with the most night votes
# #                 alive_targets = np.where(self.alive)[0]
# #                 eliminated_player = alive_targets[np.argmax(self.night_votes[alive_targets])]
# #                 role = self.roles[self.players[eliminated_player]]
# #                 log = f"    Player {eliminated_player} ({role}): eliminated by wolves"
# #                 self.episode_log.append(log)

# #                 self.alive[eliminated_player] = False
# #                 self.night_votes = np.zeros(self.num_players, dtype=int)

# #                 remaining_wolves = np.sum(self.alive & (self.players == self.roles.index('Wolf')))
# #                 remaining_villagers = np.sum(self.alive & (self.players == self.roles.index('Villager')))

# #                 if remaining_wolves == 0:
# #                     self.episode_log.append("Villagers win!")
# #                     terminated = True
# #                 elif remaining_wolves >= remaining_villagers:
# #                     self.episode_log.append("Wolves win!")
# #                     terminated = True
# #                 else:
# #                     terminated = False

# #                 # Move to day phase
# #                 self.game_phase = 0
# #                 self.current_player = 0
# #                 self.day += 1
# #                 self.episode_log.append(f"{self.day}:")
# #             else:
# #                 self.current_player += 1

# #         return self._get_observation(), reward, terminated, truncated, info


# # myenv/myenv.py
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np

# class MyEnv(gym.Env):
#     def __init__(self, num_wolves=2, num_villagers=4):
#         super(MyEnv, self).__init__()

#         # Game Parameters
#         self.num_wolves = num_wolves
#         self.num_villagers = num_villagers
#         self.num_players = self.num_wolves + self.num_villagers

#         # Roles
#         self.roles = ['Villager', 'Wolf']
#         self.role_distribution = {'Villager': self.num_villagers, 'Wolf': self.num_wolves}

#         # Observation Space
#         self.observation_space = spaces.Dict({
#             'players': spaces.MultiDiscrete([len(self.roles) + 1] * self.num_players),  # +1 for "unknown"
#             'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
#             'game_phase': spaces.Discrete(2),  # 0: Day, 1: Night
#             'my_role': spaces.Discrete(len(self.roles) + 1)  # Own role (+1 for unknown)
#         })

#         # Action Space
#         self.action_space = spaces.Discrete(self.num_players)  # Target player index

#         # Game State
#         self.players = np.zeros(self.num_players, dtype=int)  # Store role indices for each player
#         self.votes = np.zeros(self.num_players, dtype=int)
#         self.night_votes = np.zeros(self.num_players, dtype=int)
#         self.game_phase = 1  # Start with night phase
#         self.current_player = 0
#         self.alive = np.ones(self.num_players, dtype=bool)

#         # Logging
#         self.episode_log = []
#         self.day = 1

#         self.reset()

#     def reset(self, seed=None, options=None):
#         """Reset the game to the initial state."""
#         super().reset(seed=seed)
#         # Initialize roles
#         self.players = self._assign_roles()
#         self.votes = np.zeros(self.num_players, dtype=int)
#         self.night_votes = np.zeros(self.num_players, dtype=int)
#         self.game_phase = 1  # Start with night phase
#         self.current_player = 0
#         self.alive = np.ones(self.num_players, dtype=bool)
#         self.day = 1
#         self.episode_log = [f"Night {self.day}:"]

#         return self._get_observation(), {}

#     def _assign_roles(self):
#         """Assign roles to players randomly based on role distribution."""
#         roles = []
#         for role, count in self.role_distribution.items():
#             roles.extend([self.roles.index(role)] * count)
#         roles = np.random.permutation(roles)
#         return roles

#     def _get_observation(self):
#         """Get the current observation based on the current player's role."""
#         current_role = self.players[self.current_player]
#         if current_role == self.roles.index('Wolf'):
#             # Wolves see everyone's role
#             player_roles = np.where(self.alive, self.players, len(self.roles))
#         else:
#             # Villagers see everyone as "unknown" (+1 for "unknown")
#             player_roles = np.where(self.alive, len(self.roles), len(self.roles))

#         observation = {
#             'players': player_roles,
#             'votes': np.clip(self.votes, 0, self.num_players - 1),
#             'game_phase': np.clip(self.game_phase, 0, 1),
#             'my_role': current_role
#         }
#         return observation

#     def step(self, action):
#         """Take an action and return the observation, reward, terminated, truncated, and info."""
#         if action < 0 or action >= self.num_players:
#             raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}.")
#         if not self.alive[action]:
#             # Penalty for voting or attacking a dead player
#             reward = -2
#             log = f"    Player {self.current_player} voted/attacked a dead Player {action}"
#             self.episode_log.append(log)
#             terminated = False
#             truncated = False
#             return self._get_observation(), reward, terminated, truncated, {}

#         terminated = False
#         truncated = False
#         info = {}

#         if self.game_phase == 0:  # Day phase
#             # Vote to eliminate a player
#             self.votes[action] += 1
#             player_role = self.roles[self.players[self.current_player]]
#             log = f"    Player {self.current_player} ({player_role}): votes to exile Player {action}"
#             self.episode_log.append(log)

#             # Check if all players have voted
#             if self.current_player == self.num_players - 1:
#                 alive_voters = np.where(self.alive)[0]
#                 eliminated_player = alive_voters[np.argmax(self.votes[alive_voters])]

#                 role = self.roles[self.players[eliminated_player]]
#                 log = f"    Player {eliminated_player} ({role}): eliminated"
#                 self.episode_log.append(log)

#                 self.alive[eliminated_player] = False

#                 if role == 'Wolf':
#                     reward = 1
#                 else:
#                     reward = -1

#                 # Check win conditions
#                 remaining_wolves = np.sum(self.alive & (self.players == self.roles.index('Wolf')))
#                 remaining_villagers = np.sum(self.alive & (self.players == self.roles.index('Villager')))

#                 if remaining_wolves == 0:
#                     self.episode_log.append("Villagers win!")
#                     terminated = True
#                 elif remaining_wolves >= remaining_villagers:
#                     self.episode_log.append("Wolves win!")
#                     terminated = True
#                 else:
#                     terminated = False

#                 # Move to night phase
#                 self.game_phase = 1
#                 self.current_player = 0
#                 self.votes = np.zeros(self.num_players, dtype=int)
#                 self.episode_log.append("Night:")
#             else:
#                 reward = 0
#                 terminated = False
#                 self.current_player += 1
#         else:  # Night phase
#             wolves_indices = np.where((self.players == self.roles.index('Wolf')) & self.alive)[0]
#             current_role = self.players[self.current_player]

#             if self.current_player in wolves_indices:
#                 self.night_votes[action] += 1
#                 log = f"    Player {self.current_player} (Wolf): votes to attack Player {action}"
#                 self.episode_log.append(log)
#                 reward = 1
#             else:
#                 # Villagers do nothing at night
#                 log = f"    Player {self.current_player} ({self.roles[current_role]}): does nothing at night"
#                 self.episode_log.append(log)
#                 reward = 0

#             if self.current_player == self.num_players - 1:
#                 # Eliminate the player with the most night votes
#                 alive_targets = np.where(self.alive)[0]
#                 eliminated_player = alive_targets[np.argmax(self.night_votes[alive_targets])]
#                 role = self.roles[self.players[eliminated_player]]
#                 log = f"    Player {eliminated_player} ({role}): eliminated by wolves"
#                 self.episode_log.append(log)

#                 self.alive[eliminated_player] = False
#                 self.night_votes = np.zeros(self.num_players, dtype=int)

#                 remaining_wolves = np.sum(self.alive & (self.players == self.roles.index('Wolf')))
#                 remaining_villagers = np.sum(self.alive & (self.players == self.roles.index('Villager')))

#                 if remaining_wolves == 0:
#                     self.episode_log.append("Villagers win!")
#                     terminated = True
#                 elif remaining_wolves >= remaining_villagers:
#                     self.episode_log.append("Wolves win!")
#                     terminated = True
#                 else:
#                     terminated = False

#                 # Move to day phase
#                 self.game_phase = 0
#                 self.current_player = 0
#                 self.day += 1
#                 self.episode_log.append(f"Day {self.day}:")
#             else:
#                 self.current_player += 1

#         return self._get_observation(), reward, terminated, truncated, info


# myenv/myenv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MyEnv(gym.Env):
    def __init__(self, num_wolves=2, num_villagers=4):
        super(MyEnv, self).__init__()

        # Game Parameters
        self.num_wolves = num_wolves
        self.num_villagers = num_villagers
        self.num_players = self.num_wolves + self.num_villagers

        # Roles
        self.roles_names = ['Villager', 'Wolf']  # 0, 1
        self.role_distribution = {'Villager': self.num_villagers, 'Wolf': self.num_wolves}

        # Observation Space
        self.observation_space = spaces.Dict({
            'players': spaces.MultiDiscrete([len(self.roles_names) + 1] * self.num_players),  # +1 for "unknown"
            'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
            'game_phase': spaces.Discrete(2),  # 0: Day, 1: Night
            'my_role': spaces.Discrete(len(self.roles_names) + 1),  # Own role (+1 for unknown)
            'alive': spaces.MultiBinary(self.num_players)
        })

        # Action Space
        self.action_space = spaces.Discrete(self.num_players)  # Target player index

        # Game State
        self.roles = np.zeros(self.num_players, dtype=int)  # 0 代表狼 1代表村民
        self.votes = np.zeros(self.num_players, dtype=int)  # 昨天的投票
        self.night_votes = np.zeros(self.num_players, dtype=int)  # 狼人的选择
        self.game_phase = 1  # Start with night phase
        self.current_player = 0
        self.alive = np.ones(self.num_players, dtype=bool)  # 是否活着
        
        # Logging
        self.episode_log = []
        self.day = 1
        
        self.reset()
        

    def reset(self, seed=None, options=None):
        """Reset the game to the initial state."""
        super().reset(seed=seed)
        # Initialize roles
        self.roles = self._assign_roles()
        self.votes = np.zeros(self.num_players, dtype=int)
        self.night_votes = np.zeros(self.num_players, dtype=int)
        self.game_phase = 1  # Start with night phase
        self.current_player = 0
        self.alive = np.ones(self.num_players, dtype=bool)
        self.day = 1
        self.episode_log = []
        self.vote_record = {i: -1 for i in range(self.num_players)}
        self.night_vote_record = {i: -1 for i in range(self.num_players)}
        return self._get_observation(), {}

    def _assign_roles(self):
        """Assign roles to players randomly based on role distribution."""
        roles = []
        for role, count in self.role_distribution.items():
            roles.extend([self.roles_names.index(role)] * count)
        roles = np.random.permutation(roles)
        return roles

    def _get_observation(self):
        """Get the current observation based on the current player's role."""
        current_role = self.roles[self.current_player]
        if current_role == self.roles_names.index('Wolf'):
            # Wolves see everyone's role
            player_roles =self.roles
        else:
            # Villagers see everyone as "unknown" (+1 for "unknown")
            player_roles = np.array([len(self.roles_names)] * self.num_players)

        observation = {
            'players': player_roles,
            'votes': self.votes,
            'game_phase': np.clip(self.game_phase, 0, 1),
            'my_role': current_role, 
            'alive': self.alive
        }
        return observation

    def step(self, action):

        terminated = False
        truncated = False
        info = {}

        current_role = self.roles[self.current_player]

        # 白天
        if self.game_phase == 0:
            # 私人投票 和 私人被投 都不算
            if self.alive[action] == True and self.alive[self.current_player]== True:
                self.votes[action] += 1
                self.vote_record[self.current_player] = action
            
            voter_role = self.roles[self.current_player]
            log = f"    Player {self.current_player} ({voter_role}): votes to exile Player {action}"
            self.episode_log.append(log)

            # 投票完毕 进入计票环节
            if self.current_player == self.num_players - 1:

                eliminated_player = np.argmax(self.votes)
                
                role_name = self.roles_names[self.roles[eliminated_player]]
                log = f"    Player {eliminated_player} ({role_name}): eliminated"
                self.episode_log.append(log)

                self.alive[eliminated_player] = False

                reward = 0
                for k, v in self.vote_record.items():
                    if self.roles[k] == self.roles[v]:
                        if self.roles[k] == 1: # 狼人投狼人
                            reward -= 2
                        else:
                            reward -= 1
                    else:
                        reward += 1

                # 判断游戏是否结束
                remaining_wolves = np.sum(self.alive & (self.roles == self.roles_names.index('Wolf')))
                remaining_villagers = np.sum(self.alive & (self.roles == self.roles_names.index('Villager')))

                if remaining_wolves == 0:
                    self.episode_log.append("Villagers win!")
                    terminated = True
                elif remaining_villagers == 0:
                    self.episode_log.append("Wolves win!")
                    terminated = True
                else:
                    terminated = False

                # 白天结束
                self.game_phase = 1
                self.current_player = 0
                # 清空计数
                self.votes = np.zeros(self.num_players, dtype=int)
                self.vote_record = {i: -1 for i in range(self.num_players)}
                self.episode_log.append("Night:")
            
            else: # 如果投票没结束
                reward = 0
                terminated = False
                self.current_player += 1
        
        else: # 夜晚
            
            reward = 0
            if self.roles[self.current_player] == 1: 
                
                if self.alive[action] == True:
                    self.night_vote_record[self.current_player] = action
                    self.night_votes[action] += 1
                    log = f"    Player {self.current_player} (Wolf): votes to attack Player {action}"
                    self.episode_log.append(log)
                    reward = 1
                    
                else:  # 狼人投的是已经死的人
                    log = f"    Player {self.current_player} (Wolf): votes to attack Player {action} who is dead..."
                    reward -=10
            else:
                log = f"    Player {self.current_player} ({self.roles[current_role]}): does nothing at night"
                self.episode_log.append(log)
                reward = 0

            # 投票完毕 进入计票环节
            if self.current_player == self.num_players - 1:

                eliminated_player = np.argmax(self.night_votes)
                role = self.roles[eliminated_player]
                log = f"    Player {eliminated_player} ({role}): eliminated by wolves"
                self.episode_log.append(log)

                reward = 0
                for k, v in self.night_vote_record.items():
                    if self.roles[k] == self.roles[v]:
                         # 狼人攻击狼人
                        reward -= 4
                    else:
                        reward += 1
                        
                self.alive[eliminated_player] = False
                self.night_votes = np.zeros(self.num_players, dtype=int)

                remaining_wolves = np.sum(self.alive & (self.roles == self.roles_names.index('Wolf')))
                remaining_villagers = np.sum(self.alive & (self.roles == self.roles_names.index('Villager')))

                if remaining_wolves == 0:
                    self.episode_log.append("Villagers win!")
                    terminated = True
                elif remaining_villagers == 0:
                    self.episode_log.append("Wolves win!")
                    terminated = True
                else:
                    terminated = False

                # 夜晚结束
                self.game_phase = 0
                self.current_player = 0
                self.day += 1
                self.episode_log.append(f"{self.day}:")
            else:
                
                # reward = 0
                terminated = False
                self.current_player += 1

        return self._get_observation(), reward, terminated, truncated, info



class Wolf(gym.Env):
    def __init__(self, num_wolves=2, num_villagers=4):
        super(Wolf, self).__init__()

        # Game Parameters
        self.num_wolves = num_wolves
        self.num_villagers = num_villagers
        self.num_players = self.num_wolves + self.num_villagers

        # Roles
        self.roles = ['Villager', 'Wolf']
        self.role_distribution = {'Villager': self.num_villagers, 'Wolf': self.num_wolves}

        # Observation Space
        self.observation_space = spaces.Dict({
            'players': spaces.MultiDiscrete([len(self.roles) + 1] * self.num_players),  # +1 for "unknown"
            'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
            'game_phase': spaces.Discrete(2),  # 0: Day, 1: Night
            'my_role': spaces.Discrete(len(self.roles) + 1)  # Own role (+1 for unknown)
        })

        # Action Space
        self.action_space = spaces.Discrete(self.num_players)  # Target player index

        # Game State
        self.players = np.zeros(self.num_players, dtype=int)  # Store role indices for each player
        self.votes = np.zeros(self.num_players, dtype=int)
        self.night_votes = np.zeros(self.num_players, dtype=int)
        self.game_phase = 1  # Start with night phase
        self.current_player = 0
        self.alive = np.ones(self.num_players, dtype=bool)

        # Logging
        self.episode_log = []
        self.day = 1

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the game to the initial state."""
        super().reset(seed=seed)
        # Initialize roles
        self.players = self._assign_roles()
        self.votes = np.zeros(self.num_players, dtype=int)
        self.night_votes = np.zeros(self.num_players, dtype=int)
        self.game_phase = 1  # Start with night phase
        self.current_player = 0
        self.alive = np.ones(self.num_players, dtype=bool)
        self.day = 1
        self.episode_log = [f"Night {self.day}:"]

        return self._get_observation(), {}

    def _assign_roles(self):
        roles = []
        for role, count in self.role_distribution.items():
            roles.extend([self.roles.index(role)] * count)
        roles = np.random.permutation(roles)
        return roles

    def _get_observation(self):
        """Get the current observation based on the current player's role."""
        current_role = self.players[self.current_player]
        if current_role == self.roles.index('Wolf'):
            # Wolves see everyone's role
            player_roles = np.where(self.alive, self.players, len(self.roles))
        else:
            # Villagers see everyone as "unknown" (+1 for "unknown")
            player_roles = np.where(self.alive, len(self.roles), len(self.roles))

        observation = {
            'players': player_roles,
            'votes': np.clip(self.votes, 0, self.num_players - 1),
            'game_phase': np.clip(self.game_phase, 0, 1),
            'my_role': current_role,
            # 'alive': self.alive
        }
        return observation

    def step(self, action):
        """Take an action and return the observation, reward, terminated, truncated, and info."""
        if action < 0 or action >= self.num_players:
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_players - 1}.")
        # if not self.alive[action]:
        #     # Penalty for voting or attacking a dead player
        #     reward = -2
        #     log = f"    Player {self.current_player} voted/attacked a dead Player {action}"
        #     self.episode_log.append(log)
        #     terminated = False
        #     truncated = False
        #     return self._get_observation(), reward, terminated, truncated, {}

        terminated = False
        truncated = False
        info = {}

        current_role = self.players[self.current_player]

        if self.game_phase == 0:  # Day phase
            # Vote to eliminate a player
            self.votes[action] += 1
            player_role = self.roles[self.players[self.current_player]]
            log = f"    Player {self.current_player} ({player_role}): votes to exile Player {action}"
            self.episode_log.append(log)

            # Check if all players have voted
            if self.current_player == self.num_players - 1:
                alive_voters = np.where(self.alive)[0]
                eliminated_player = alive_voters[np.argmax(self.votes[alive_voters])]

                role = self.roles[self.players[eliminated_player]]
                log = f"    Player {eliminated_player} ({role}): eliminated"
                self.episode_log.append(log)

                self.alive[eliminated_player] = False

                if role != current_role:
                    reward = 1
                else:
                    reward = -1

                # Check win conditions
                remaining_wolves = np.sum(self.alive & (self.players == self.roles.index('Wolf')))
                remaining_villagers = np.sum(self.alive & (self.players == self.roles.index('Villager')))

                if remaining_wolves == 0:
                    self.episode_log.append("Villagers win!")
                    terminated = True
                elif remaining_wolves >= remaining_villagers:
                    self.episode_log.append("Wolves win!")
                    terminated = True
                else:
                    terminated = False

                # Move to night phase
                self.game_phase = 1
                self.current_player = 0
                self.votes = np.zeros(self.num_players, dtype=int)
                self.episode_log.append("Night:")
            else:
                reward = 0
                terminated = False
                self.current_player += 1
        else:  # Night phase
            wolves_indices = np.where((self.players == self.roles.index('Wolf')) & self.alive)[0]

            if self.current_player in wolves_indices:
                self.night_votes[action] += 1
                log = f"    Player {self.current_player} (Wolf): votes to attack Player {action}"
                self.episode_log.append(log)
            else:
                # Villagers do nothing at night
                log = f"    Player {self.current_player} ({self.roles[current_role]}): does nothing at night"
                self.episode_log.append(log)

            if self.current_player == self.num_players - 1:
                # Eliminate the player with the most night votes
                alive_targets = np.where(self.alive)[0]
                eliminated_player = alive_targets[np.argmax(self.night_votes[alive_targets])]
                role = self.roles[self.players[eliminated_player]]
                log = f"    Player {eliminated_player} ({role}): eliminated by wolves"
                self.episode_log.append(log)

                
                if role != current_role:
                    reward = 1
                else:
                    reward = -1

                self.alive[eliminated_player] = False
                self.night_votes = np.zeros(self.num_players, dtype=int)

                remaining_wolves = np.sum(self.alive & (self.players == self.roles.index('Wolf')))
                remaining_villagers = np.sum(self.alive & (self.players == self.roles.index('Villager')))

                if remaining_wolves == 0:
                    self.episode_log.append("Villagers win!")
                    terminated = True
                elif remaining_wolves >= remaining_villagers:
                    self.episode_log.append("Wolves win!")
                    terminated = True
                else:
                    terminated = False

                # Move to day phase
                self.game_phase = 0
                self.current_player = 0
                self.day += 1
                self.episode_log.append(f"{self.day}:")
            else:
                
                reward = 0
                terminated = False
                self.current_player += 1

        return self._get_observation(), reward, terminated, truncated, info
