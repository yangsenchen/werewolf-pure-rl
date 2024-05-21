import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MyEnv(gym.Env):
    def __init__(self, num_wolves=3, num_villagers=6, num_seer=0, num_hunter=0, num_witch=0, rival=None, camp=0, debug_mode=False):
        super(MyEnv, self).__init__()
        self.debug_mode = debug_mode
        
        # 对手
        self.rival = rival

        # 角色人数
        self.num_wolves = num_wolves
        self.num_villagers = num_villagers
        self.num_seer = num_seer
        self.num_hunter = num_hunter
        self.num_witch = num_witch
        self.num_players = self.num_wolves + self.num_villagers + self.num_seer + self.num_hunter + self.num_witch

        # 角色分配  0: Villager, 1: Wolf, 2: Seer, 3: Witch, 4: Hunter
        self.roles_names = ['村民', '狼人', '卜师', '女巫', '猎人'] 
        self.role_distribution = {'村民': self.num_villagers, '狼人': self.num_wolves, '卜师': self.num_seer, '女巫': self.num_witch, '猎人': self.num_hunter}
        self.seer_known_roles = np.zeros(self.num_players, dtype=int)
        self.witch_known_roles = np.zeros(self.num_players, dtype=int)
        
        # 当前模型 train的是哪个阵营  0: 村民 以及 神职, 1: 狼人
        self.who_is_training = camp
        
        # Observation Space
        self.observation_space = spaces.Dict({
            'players': spaces.MultiDiscrete([len(self.roles_names) + 1] * self.num_players),  # +1 for "unknown"
            'votes': spaces.MultiDiscrete([self.num_players] * self.num_players),
            'game_phase': spaces.Discrete(2),  # 0: Day, 1: Night
            'my_role': spaces.Discrete(len(self.roles_names) + 1),  # Own role (+1 for unknown)
            'alive': spaces.MultiBinary(self.num_players)
        })  

        # Action Space
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)  # Action space now includes role guesses and a target player index

        # Game State
        self.roles = np.zeros(self.num_players, dtype=int)  # 0代表村民 1代表狼人
        self.votes = np.zeros(self.num_players, dtype=int)  # 昨天的投票
        self.night_votes = np.zeros(self.num_players, dtype=int)  # 狼人的选择
        self.day_or_night = 1  # 从夜晚开始 0
        self.current_player = 0 # 当前玩家的身份 
        self.alive = np.ones(self.num_players, dtype=bool)  # 是否活着
        
        # Logging
        self.log = ["第1天 夜晚:"]
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
        self.log = ["第1天 夜晚:"]
        
        return self._get_observation(self.who_is_training), {}

    def _random_assign_roles(self): 
        """Assign roles to players randomly based on role distribution. return is like [0,1,0,1,0,1,1,1]"""
        roles = []
        for role, count in self.role_distribution.items():
            roles.extend([self.roles_names.index(role)] * count)
        roles = np.random.permutation(roles)
        return roles

    def _get_observation(self, current_role):
        """Get the current observation based on the current player's role."""
        
        if current_role == self.roles_names.index('狼人'):
            # 狼人看得到所有玩家身份
            player_roles =self.roles
        else:
            # 村民不知道任何身份信息 Villagers see everyone as "unknown" (+1 for "unknown")
            player_roles = np.array([len(self.roles_names)] * self.num_players)

        observation = {
            'players': player_roles,  # 玩家角色
            'votes': self.votes,   # 投票信息
            'game_phase': np.clip(self.day_or_night, 0, 1),  # 现在是白天或黑夜？ 这个检查一下
            'my_role': current_role,  # 我的身份
            'alive': self.alive  # 活着的玩家
        }
        # print(observation)
        return observation

    def step(self, action):
        # Extract guessed roles and target player index from the action
        guessed_roles = action[:-1]
        target_player = int(action[-1] * self.num_players)-1  # Scale to player index

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # 当前的角色身份
        current_role = self.roles[self.current_player]
        # 当前的角色名字
        current_role_name = self.roles_names[current_role]
        # 当前的角色是否活着
        current_role_alive = self.alive[self.current_player]
        
        if current_role != self.who_is_training:
            # 获得的是其他玩家的动作捏
            action, _states = self.rival.predict(self._get_observation(current_role), deterministic=True)
            guessed_roles = action[:-1]
            target_player = int(action[-1] * self.num_players)-1  # Scale to player index
            
        # 白天
        if self.day_or_night == 0:
            if current_role_alive:
                # 计票环节 投给死人不算数
                if self.alive[target_player] == 1:
                    self.votes[target_player] += 1
                    reward += 10
                    self.log.append(f"    P{self.current_player} ({current_role_name}): 投给 P{target_player} ({self.roles_names[self.roles[target_player]]})")
                else: #  惩罚投给死人的行为
                    reward += -50
                    self.log.append(f"    P{self.current_player} ({current_role_name}): 投给 P{target_player} 但是这个人其实已经死了...")
            
            # 白天的投票完毕 进入计票环节
            if self.current_player == self.num_players - 1:
                exiled_player = np.argmax(self.votes)
                self.alive[exiled_player] = 0

                self.log.append(f"    P{exiled_player} ({self.roles_names[self.roles[exiled_player]]}): 出局")

                # 如果说投票结束之后 己方阵营死人了 就是reward为负
                if self.roles[exiled_player] == self.who_is_training:
                    reward += -10
                else:
                    reward += 10
                
                # 判断游戏是否结束
                r, terminated = self._is_game_over()    
                reward += r

                # 白天结束
                self.day_or_night = 1
                self.current_player = 0
                self.log.append(f"第{self.day}天 夜晚: 活人{str([i for i in range(len(self.alive)) if self.alive[i] == 1])}") # 打印剩余的玩家

                # 清空计数
                self.votes = np.zeros(self.num_players, dtype=int)
            
            else: # 如果投票没结束
                terminated = False
                self.current_player += 1
        
        else: # 夜晚
            # 当前的角色是狼人
            if current_role == 1: 
                if current_role_alive:
                    # 如果当前选择攻击的角色是活着的 票才是有效的
                    if self.alive[target_player] == 1:
                        self.night_votes[target_player] += 1
                        self.log.append(f"    P{self.current_player} ({current_role_name}): 想杀 P{target_player}")
                        if self.who_is_training == 1: # 训练狼人的时候才有reward 村民的话 晚上是没有reward的
                            reward += 1
                    else:  # 狼人想杀死的是已经死的人
                        self.log.append(f"    P{self.current_player} ({current_role_name}): 想杀的P{target_player}已经死了...")
                        if self.who_is_training == 1: 
                            reward += -50
            else: # 当前角色是村民 啥也不干 睡觉
                if current_role_alive:
                    self.log.append(f"    P{self.current_player} ({current_role_name}): 睡觉")

            # 狼人内部投票完毕 进入计票环节
            if self.current_player == self.num_players - 1:
                # 死人
                killed_player = np.argmax(self.night_votes)
                self.alive[killed_player] = 0

                self.log.append(f"    P{killed_player} ({self.roles_names[self.roles[killed_player]]}): 被杀")
                
                # 狼人投票结束 重置
                self.night_votes = np.zeros(self.num_players, dtype=int)

                r, terminated = self._is_game_over()
                reward += r
                
                # 夜晚结束
                self.day_or_night = 0 # 进入白天
                self.current_player = 0 # 白天从第一个玩家开始发言
                self.day += 1 # 天数加1
                self.log.append(f"第{self.day}天 白天: 活人{str([i for i in range(len(self.alive)) if self.alive[i] == 1])}") # 打印剩余的玩家
            else: # 如果投票没结束
                terminated = False
                self.current_player += 1

        if terminated:
            for log in self.log:
                print(log)
            # print("Game Over")
            print("\n")
            print("\n")
            print("\n")
            
        return self._get_observation(self.who_is_training), reward, terminated, truncated, info


    def _is_game_over(self):
        # 判断游戏是否结束
        remaining_wolves = np.sum(self.alive & (self.roles == self.roles_names.index('狼人')))
        remaining_villagers = np.sum(self.alive & (self.roles == self.roles_names.index('村民')))
        self.log.append(f"    剩余狼人: {remaining_wolves}")
        self.log.append(f"    剩余村民: {remaining_villagers}")
        # 如果狼人死光了
        if remaining_wolves == 0:
            self.log.append("村民胜利!")
            if self.who_is_training == 0:
                reward = 10
            else:
                reward = -10
            terminated = True
        # 如果村民死光了
        elif remaining_villagers == 0:
            self.log.append("狼人胜利!")
            if self.who_is_training == 1:
                reward = 10
            else:
                reward = -10
            terminated = True
        else:
            reward = 0
            terminated = False

        return reward, terminated
