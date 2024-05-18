
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
        self.current_player = 0 # 当前玩家的身份
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
            # 死人投票 和 死人被投 都不算
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


