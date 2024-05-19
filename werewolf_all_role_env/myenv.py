
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MyEnv(gym.Env):
    def __init__(self, num_villagers=3, num_wolves=3, num_witch=1, num_hunter=1, num_seer =1, rival= None, camp=0, debug_mode=False):
        super(MyEnv, self).__init__()
        self.debug_mode = debug_mode
        # Game Parameters
        self.num_wolves = num_wolves
        self.num_villagers = num_villagers
        self.num_players = self.num_wolves + self.num_villagers
        self.rival = rival
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

        # who to train in this game
        self.who_to_train = camp
        # Action Space
        self.action_space = spaces.Discrete(self.num_players)  # Target player index

        # Game State
        self.roles = np.zeros(self.num_players, dtype=int)  # 0代表村民 1代表狼人
        self.votes = np.zeros(self.num_players, dtype=int)  # 昨天的投票
        self.night_votes = np.zeros(self.num_players, dtype=int)  # 狼人的选择
        self.day_or_night = 1  # 从夜晚开始 0
        self.current_player = 0 # 当前玩家的身份 
        self.alive = np.ones(self.num_players, dtype=bool)  # 是否活着
        
        # Logging
        self.log = []
        self.day = 1
        
        self.reset()
        

    def reset(self, seed=None, options=None):
        """Reset the game to the initial state. 只会在一局游戏的最开始调用"""
        super().reset(seed=seed)
        # Initialize roles
        self.roles = self._random_assign_roles()
        self.votes = np.zeros(self.num_players, dtype=int)
        self.night_votes = np.zeros(self.num_players, dtype=int)
        self.day_or_night = 1  # 从夜晚开始
        self.current_player = 0 # 当前玩家编号
        self.alive = np.ones(self.num_players, dtype=bool)
        self.day = 1
        self.log = []
        
        return self._get_observation(self.who_to_train), {}

    def _random_assign_roles(self): 
        """Assign roles to players randomly based on role distribution. return is like [0,1,0,1,0,1,1,1]"""
        roles = []
        for role, count in self.role_distribution.items():
            roles.extend([self.roles_names.index(role)] * count)
        roles = np.random.permutation(roles)
        return roles

    def _get_observation(self, current_role):
        """Get the current observation based on the current player's role."""
        
        if current_role == self.roles_names.index('Wolf'):
            # 狼人看得到所有玩家身份
            player_roles =self.roles
        else:
            # 村民不知道 Villagers see everyone as "unknown" (+1 for "unknown")
            player_roles = np.array([len(self.roles_names)] * self.num_players)

        observation = {
            'players': player_roles,
            'votes': self.votes,
            'game_phase': np.clip(self.day_or_night, 0, 1),
            'my_role': current_role, 
            'alive': self.alive
        }
        # print observation in one line
        # if self.debug_mode:
        #     print(f"Observation: {observation}")
        return observation

    def step(self, action):
        
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # 当前的角色身份
        current_role = self.roles[self.current_player]
        current_role_name = self.roles_names[current_role]

        if current_role!= self.who_to_train:
            # 获得的是对面的动作
            action, _states = self.rival.predict(self._get_observation(current_role), deterministic=True)
            
            
        # 白天
        if self.day_or_night == 0:
            
            # 计票环节 (死人投票 和 死人被投 都不算)
            if self.alive[action] == 1 and self.alive[self.current_player]== 1:
                self.votes[action] += 1
            
            self.log.append(f"    Player {self.current_player} ({current_role_name}): votes to exile Player {action}")
            
            # 白天的投票完毕 进入计票环节
            if self.current_player == self.num_players - 1:
                exiled_player = np.argmax(self.votes)
                self.alive[exiled_player] = 0

                self.log.append(f"    Player {exiled_player} ({current_role_name}): exiled")

                # 如果说投票结束之后 己方阵营死人了 就是reward为负
                if self.roles[exiled_player] == self.who_to_train:
                    reward -= 10
                else:
                    reward += 10
                
                # 判断游戏是否结束
                r, terminated = self._is_game_over()    
                reward += r

                # 白天结束
                self.day_or_night = 1
                self.current_player = 0
                # 清空计数
                self.votes = np.zeros(self.num_players, dtype=int)
            
            else: # 如果投票没结束
                terminated = False
                self.current_player += 1
        
        else: # 夜晚
            
            # 当前训练的角色是狼人
            if self.who_to_train == 1: 
                # 如果当前选择攻击的角色是活着的 并且当前狼人本身是活着的 票才是有效的
                if self.alive[action] == 1 and self.alive[self.current_player]== 1:
                    self.night_votes[action] += 1
                    log = f"    P{self.current_player} ({current_role_name}): votes to attack P{action}"
                    self.log.append(log)
                    reward = 1

                else:  # 狼人想杀死的是已经死的人
                    log = f"    P{self.current_player} ({current_role_name}): votes to attack P{action} who is dead..."
                    self.log.append(log)
                    reward -=10

            else: # 当前训练角色是村民
                reward = 0
                log = f"    P{self.current_player} ({current_role_name}): sleeps..."
                self.log.append(log)

            # 狼人内部投票完毕 进入计票环节
            if self.current_player == self.num_players - 1:
                killed_player = np.argmax(self.night_votes)
                log = f"    P{killed_player} ({current_role_name}): is killed"
                self.log.append(log)
                reward = 0
                self.alive[killed_player] = 0
                self.night_votes = np.zeros(self.num_players, dtype=int)

                r, terminated = self._is_game_over()
                reward += r
                
                # 夜晚结束
                self.day_or_night = 0 # 进入白天
                self.current_player = 0 # 白天从第一个玩家开始发言
                self.day += 1 # 天数加1

            else: # 如果投票没结束
                terminated = False
                self.current_player += 1

        if terminated and self.debug_mode:
            for log in self.log:
                print(log)
            print("Game Over")
            print("\n")
            print("\n")
            print("\n")
            
        return self._get_observation(self.who_to_train), reward, terminated, truncated, info


    def _is_game_over(self):
        
        reward = 0
        # 判断游戏是否结束
        remaining_wolves = np.sum(self.alive & (self.roles == self.roles_names.index('Wolf')))
        remaining_villagers = np.sum(self.alive & (self.roles == self.roles_names.index('Villager')))
        self.log.append(f"    Remaining wolves: {remaining_wolves}")
        self.log.append(f"    Remaining villagers: {remaining_villagers}")
        if remaining_wolves == 0:
            self.log.append("Villagers win!")
            if self.who_to_train == 0:
                reward += 10
            else:
                reward -= 10
            terminated = True
        elif remaining_villagers == 0:
            self.log.append("Wolves win!")
            if self.who_to_train == 1:
                reward += 10
            else:
                reward -= 10
            terminated = True
        else:
            reward = 0
            terminated = False
            self.log.append(f"{self.day} Daytime:")

        return reward, terminated