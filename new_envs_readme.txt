# 新建环境的reward说明

#### Atari
  ## dir : envs/atari/

  ## Freeway
     # 新增文件: envs/atari/atari_freeway_env.py
     # 修改文件: envs/atari/__init__.py    # 在此文件导入atari_freeway_env.py模块
                 envs/__init__.py          # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2])
       reward-1 : 原始的reward
       reward-2 : 小鸡高度改变量
     # 最终环境名:
       FreewayNoFrameskip-v0               # 原始gym提供的reward的环境
       FreewayNoFrameskip-v0-reward-0      # np.array([reward-1, reward-2])
       FreewayNoFrameskip-v0-reward-1      # 原始reward
       FreewayNoFrameskip-v0-reward-2      # 自己修改reward得到的环境
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = 0.0, max = 1.0, max-abs = 1.0
       reward-2 : min = -10.0, max = 10.0, max-abs = 10.0

  ## Atlantis
     # 新增文件: envs/atari/atari_atlantis_env.py
     # 修改文件: envs/atari/__init__.py    # 在此文件导入atari_atlantis_env.py模块
                envs/__init__.py          # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2, reward-3])
       reward-1 : 原始的reward
       reward-2 : 每走一步得1分
       reward-3 : 丢失掉一条命-1分
     # 最终环境名:
       AtlantisNoFrameskip-v0               # 原始gym提供的reward的环境
       AtlantisNoFrameskip-v0-reward-0      # np.array([reward-1, reward-2])
       AtlantisNoFrameskip-v0-reward-1      # 原始的reward
       AtlantisNoFrameskip-v0-reward-2      # (210, 160, 3) + reward-2
       AtlantisNoFrameskip-v0-reward-3      # reward-3
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = 0.0, max = 5000.0, max-abs = 5000.0
       reward-2 : min = 0.0, max = 1.0, max-abs = 1.0
       reward-3 : min = -5.0, max = 0.0, max-abs = 5.0

  ## SpaceInvaders
     # 新增文件: envs/atari/atari_spaceinvaders_env.py
     # 修改文件: envs/atari/__init__.py    # 在此文件导入atari_spaceinvaders_env.py模块
                 envs/__init__.py          # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2])
       reward-1 : 原始的reward
       reward-2 : 每走一步得1分,丢失掉一条命的话-5分
     # 最终环境名:
       SpaceInvadersNoFrameskip-v0               # 原始gym提供的reward的环境
       SpaceInvadersNoFrameskip-v0-reward-0      # np.array([reward-1, reward-2])
       SpaceInvadersNoFrameskip-v0-reward-1      # 原始的reward
       SpaceInvadersNoFrameskip-v0-reward-2      # 自己修改reward得到的环境
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = 0.0, max = 30.0, max-abs = 30.0
       reward-2 : min = -5.0, max = 1.0, max-abs = 5.0

  ## PS:
     # 为了使得新增的环境名得到注册,需要修改部分envs/registration.py文件
     # 在envs/__init__.py里面注册了很多可用的环境名，比如:
         Freeway-v0-reward-2
         FreewayDeterministic-v0-reward-2
         FreewayRAM-v0-reward-2
       但是在baselines里面必须使用带有NoFrameskip的环境
     # 默认环境给出的状态都是(210, 160, 3)的图片，除了带有RAM的环境名
     # XXX-v0 等价于 XXX-v0-reward-0




#### Classic Control
  ## dir : envs/classic_control/

  ## CartPole
     # 新增文件: envs/classic_control/cartpole_reward.py
     # 修改文件: envs/classic_control/__init__.py    # 在此文件导入cartpole_reward.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2, reward-3])
       reward-1 : 原始的reward，倒下为0
       reward-2 : 位置的偏移量，离中间越近越高
       reward-3 : 角度的偏移量，角度偏离越小越高
     # 最终环境名:
       CartPole-v0                          # 原始gym提供的reward的环境
       CartPoleRAM-v0-reward-0              # (x, x_dot, theta, theta_dot) + np.array([reward-1, reward-2, reward-3])
       CartPoleRAM-v0-reward-1              # (x, x_dot, theta, theta_dot) + reward-1
       CartPoleRAM-v0-reward-2              # (x, x_dot, theta, theta_dot) + reward-2
       CartPoleRAM-v0-reward-3              # (x, x_dot, theta, theta_dot) + reward-3
       CartPoleImage-v0-reward-0            # image (84, 84, 3) + np.array([reward-1, reward-2, reward-3])
       CartPoleImage-v0-reward-1            # image (84, 84, 3) + reward-1
       CartPoleImage-v0-reward-2            # image (84, 84, 3) + reward-2
       CartPoleImage-v0-reward-3            # image (84, 84, 3) + reward-3
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = 0.0, max = 1.0, max-abs = 1.0
       reward-2 : min = -0.05, max = 0.05, max-abs = 0.05
       reward-3 : min = -0.1, max = 0.1, max-abs = 0.1

  ## MountainCar
     # 新增文件: envs/classic_control/mountain_car_reward.py
     # 修改文件: envs/classic_control/__init__.py    # 在此文件导入mountain_car_reward.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1,...,reward-5])
       reward-1 : 原始的reward，到终点获得得分
       reward-2 : 能量, gh + 0.5 * v * v, 当g=9.8时，二者量级差不多
       reward-3 : 距离终点距离
       reward-4 : 速度
       reward-5 : 高度
     # 最终环境名:
       MountainCar-v0                          # 原始gym提供的reward的环境
       MountainCarRAM-v0-reward-0              # (position, speed) + np.array([reward-1,...,reward-5])
       MountainCarRAM-v0-reward-1              # (position, speed) + reward-1
       MountainCarRAM-v0-reward-2              # (position, speed) + reward-2
       MountainCarRAM-v0-reward-3              # (position, speed) + reward-3
       MountainCarRAM-v0-reward-4              # (position, speed) + reward-4
       MountainCarRAM-v0-reward-5              # (position, speed) + reward-5
       MountainCarImage-v0-reward-0            # image (84, 84, 3) + np.array([reward-1,...,reward-5])
       MountainCarImage-v0-reward-1            # image (84, 84, 3) + reward-1
       MountainCarImage-v0-reward-2            # image (84, 84, 3) + reward-2
       MountainCarImage-v0-reward-3            # image (84, 84, 3) + reward-3
       MountainCarImage-v0-reward-4            # image (84, 84, 3) + reward-4
       MountainCarImage-v0-reward-5            # image (84, 84, 3) + reward-5
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = -1.0, max = 0.0, max-abs = 1.0
       reward-2 : min = -0.8, max = 0.8, max-abs = 0.8
       reward-3 : min = -0.09, max = 0.09, max-abs = 0.09
       reward-4 : min = -0.01, max = 0.01, max-abs = 0.01
       reward-5 : min = -0.1, max = 0.1, max-abs = 0.1

  ## Acrobot
     # 新增文件: envs/classic_control/acrobot_reward.py
     # 修改文件: envs/classic_control/__init__.py    # 在此文件导入acrobot_reward.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2])
       reward-1 : 原始的reward
       reward-2 : 高度
     # 最终环境名:
       Acrobot-v1                          # 原始gym提供的reward的环境
       AcrobotRAM-v1-reward-0              # (cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot) + np.array([reward-1, reward-2])
       AcrobotRAM-v1-reward-1              # (cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot) + reward-1
       AcrobotRAM-v1-reward-2              # (cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot) + reward-2
       AcrobotImage-v1-reward-0            # image (84, 84, 3) + np.array([reward-1, reward-2])
       AcrobotImage-v1-reward-1            # image (84, 84, 3) + reward-1
       AcrobotImage-v1-reward-2            # image (84, 84, 3) + reward-2
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = -1.0, max = 0.0, max-abs = 1.0
       reward-2 : min = -2.0, max = 2.0, max-abs = 2.0


#### PLE environment(PyGame Learning Environment)
  ## 背景说明
     Atari小游戏是基于ALE开发的，但是Atari小游戏并不会返回一些内在状态信息，比如坐标位置等等
     在PLE中也有几个小游戏，包括Catcher, Monster Kong, FlappyBird, Pixelcopter, Pong, PuckWorld, RaycastMaze, Snake, WaterWorld
     下面就是如何修改PLE提供的小游戏的reward，具体包括两个步骤:
        (1) 如何将这些小游戏统一到gym中，即通过gym.make("FlappyBird-v0")来调用的类型
        (2) 修改reward

  ## gym中调用PLE
     # refer link1:
       https://github.com/openai/gym/tree/master/gym/envs  # gym中如何新建环境
     # refer link2:
       https://pygame-learning-environment.readthedocs.io/en/latest/user/games/monsterkong.html  # PLE官网
     # refer link3:
       https://github.com/lusob/gym-ple   # gym 与 ple的重要桥梁

     # step1: 安装PLE, 参考link2左侧Home/Installation
       $ git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
       $ cd PyGame-Learning-Environment
       $ pip install -e .

     # step2: 安装gym-ple, 参考link3
       $ git clone https://github.com/lusob/gym-ple.git
       $ cd gym-ple/
       $ pip install -e .

     # step3: 融合
       # 复制 gym-ple/gym_ple 文件夹到 gym/envs/ 下面
         $ mv gym-ple/gym_ple XXX/gym/envs/
       # 现在进入到gym/envs/gym_ple目录下
         $ cd gym/envs.gym_ple/
       # 将__init__.py里面for循环注册的部分剪切到gym/envs/__init__.py中
         修改一下entry_point路径: entry_point='gym.envs.gym_ple:PLEEnv'
       # 然后将__init__.py里面全部删掉
         只保留一行: from gym.envs.gym_ple.ple_env import PLEEnv

     # step4: 收尾
       此时就可以把安装的gym-ple删除掉了，已经用不到了
       (一定要删掉不然注册游戏名字时会冲突)

  ## 测试
     # 现在可以在gym中直接import gym，然后gym.make("FlappyBird-v0")了

  ## getGameState()
     # 为了修改reward，需要调用getGameState()获得程序的内在信息
     # 直接调用会报错
     # 需要修改gym/envs/gym_ple/ple_env.py文件
     # 具体修改见文件
     # 参考链接: https://pygame-learning-environment.readthedocs.io/en/latest/user/tutorial/non_visual_state.html

  ## dir : envs/gym_ple/

  ## Catcher
     # 新增文件: envs/gym_ple/catcher_ple_env.py
     # 修改文件: envs/gym_ple/__init__.py            # 在此文件导入catcher_ple_env.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2])
       reward-1 : 原始的reward
       reward-2 : 挡板与物体水平距离的改变量
     # 最终环境名:
       Catcher-v0                                    # 原始gym提供的reward的环境
       CatcherRAM-v0-reward-0                        # getGameState() + np.array([reward-1, reward-2])
       CatcherRAM-v0-reward-1                        # getGameState() + reward-1
       CatcherRAM-v0-reward-2                        # getGameState() + reward-2
       CatcherImage-v0-reward-0                      # image (84, 84, 3) + np.array([reward-1, reward-2])
       CatcherImage-v0-reward-1                      # image (84, 84, 3) + reward-1
       CatcherImage-v0-reward-2                      # image (84, 84, 3) + reward-2
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = -1.0, max = 1.0, max-abs = 1.0
       reward-2 : min = -2.0, max = 2.0, max-abs = 2.0

  ## FlappyBird
     # 新增文件: envs/gym_ple/flappy_bird_ple_env.py
     # 修改文件: envs/gym_ple/__init__.py            # 在此文件导入flappy_bird_ple_env.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2])
       reward-1 : 原始的reward
       reward-2 : 与下一个管道的中间位置的竖直距离的改变量
     # 最终环境名:
       FlappyBird-v0                                    # 原始gym提供的reward的环境
       FlappyBirdRAM-v0-reward-0                        # getGameState() + np.array([reward-1, reward-2])
       FlappyBirdRAM-v0-reward-1                        # getGameState() + reward-1
       FlappyBirdRAM-v0-reward-2                        # getGameState() + reward-2
       FlappyBirdImage-v0-reward-0                      # image (84, 84, 3) + np.array([reward-1, reward-2])
       FlappyBirdImage-v0-reward-1                      # image (84, 84, 3) + reward-1
       FlappyBirdImage-v0-reward-2                      # image (84, 84, 3) + reward-2
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = -5.0, max = 1.0, max-abs = 5.0
       reward-2 : min = -10.0, max = 10.0, max-abs = 10.0

  ## WaterWorld
     # 新增文件: envs/gym_ple/water_world_ple_env.py
     # 修改文件: envs/gym_ple/__init__.py            # 在此文件导入water_world_ple_env.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : np.array([reward-1, reward-2])
       reward-1 : 原始的reward
       reward-2 : dis = mean(dis_from_good) - mean(dis_from_bad) 的改变量
     # 最终环境名:
       WaterWorld-v0                                    # 原始gym提供的reward的环境
       WaterWorldImage-v0-reward-0                      # image (84, 84, 3) + np.array([reward-1, reward-2])
       WaterWorldImage-v0-reward-1                      # image (84, 84, 3) + reward-1
       WaterWorldImage-v0-reward-2                      # image (84, 84, 3) + reward-2
     # Reward scaling: Normalize divide by max-abs value
       reward-1 : min = -1.0, max = 10, max-abs = 10
       reward-2 : min = -5.0, max = 5.0, max-abs = 5.0

     # PS:
       不提供RAM状态
