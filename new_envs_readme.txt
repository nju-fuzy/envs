# 新建环境的reward说明

#### Atari
  ## dir : envs/atari/

  ## Freeway
     # 新增文件: envs/atari/atari_freeway_env.py
     # 修改文件: envs/atari/__init__.py    # 在此文件导入atari_freeway_env.py模块
                 envs/__init__.py          # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : [reward-1, reward-2]
       reward-1 : 原始的reward
       reward-2 : 小鸡高度改变量
     # 最终环境名:
       FreewayNoFrameskip-v0               # 原始gym提供的reward的环境
       FreewayNoFrameskip-v0-reward-0      # [reward-1, reward-2]
       FreewayNoFrameskip-v0-reward-1      # 原始reward
       FreewayNoFrameskip-v0-reward-2      # 自己修改reward得到的环境

  ## Atlantis
     # 新增文件: envs/atari/atari_atlantis_env.py
     # 修改文件: envs/atari/__init__.py    # 在此文件导入atari_atlantis_env.py模块
                envs/__init__.py          # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : [reward-1, reward-2]
       reward-1 : 原始的reward
       reward-2 : 每走一步得0.1分,丢失掉一条命的话失掉5分
     # 最终环境名:
       AtlantisNoFrameskip-v0               # 原始gym提供的reward的环境
       AtlantisNoFrameskip-v0-reward-0      # [reward-1, reward-2]
       AtlantisNoFrameskip-v0-reward-1      # 原始的reward
       AtlantisNoFrameskip-v0-reward-2      # 自己修改reward得到的环境

  ## SpaceInvaders
     # 新增文件: envs/atari/atari_spaceinvaders_env.py
     # 修改文件: envs/atari/__init__.py    # 在此文件导入atari_spaceinvaders_env.py模块
                 envs/__init__.py          # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : [reward-1, reward-2]
       reward-1 : 原始的reward
       reward-2 : 每走一步得0.1分,丢失掉一条命的话失掉5分
     # 最终环境名:
       SpaceInvadersNoFrameskip-v0               # 原始gym提供的reward的环境
       SpaceInvadersFrameskip-v0-reward-0      # [reward-1, reward-2]
       SpaceInvadersFrameskip-v0-reward-1      # 原始的reward
       SpaceInvadersNoFrameskip-v0-reward-2      # 自己修改reward得到的环境
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
       reward-0 : [reward-1, reward-2, reward-3]
       reward-1 : 原始的reward，倒下为0
       reward-2 : 位置的偏移量，离中间越近越高
       reward-3 : 角度的偏移量，角度偏离越小越高
     # 最终环境名:
       CartPole-v0                          # 原始gym提供的reward的环境
       CartPoleRAM-v0-reward-0              # (x, x_dot, theta, theta_dot) + [reward-1, reward-2, reward-3]
       CartPoleRAM-v0-reward-1              # (x, x_dot, theta, theta_dot) + reward-1
       CartPoleRAM-v0-reward-2              # (x, x_dot, theta, theta_dot) + reward-2
       CartPoleRAM-v0-reward-3              # (x, x_dot, theta, theta_dot) + reward-3
       CartPoleImage-v0-reward-0            # image (210, 160, 3) + [reward-1, reward-2, reward-3]
       CartPoleImage-v0-reward-1            # image (210, 160, 3) + reward-1
       CartPoleImage-v0-reward-2            # image (210, 160, 3) + reward-2
       CartPoleImage-v0-reward-3            # image (210, 160, 3) + reward-3

     # PS:
       # 自己定义的CartPole要指定使用RAM还是Image，使用RAM的话是默认返回四元组
         Image是对图像从(400, 600, 3) resize 到 (160, 210, 3) 再转置
       # 定义的Image的目前还不能直接使用baselines训练

  ## MountainCar
     # 新增文件: envs/classic_control/mountain_car_reward.py
     # 修改文件: envs/classic_control/__init__.py    # 在此文件导入mountain_car_reward.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : [reward-1,...,reward-5]
       reward-1 : 原始的reward，到终点获得得分
       reward-2 : 能量, gh + 0.5 * v * v
       reward-3 : 距离终点距离
       reward-4 : 速度
       reward-5 : 高度
     # 最终环境名:
       MountainCar-v0                          # 原始gym提供的reward的环境
       MountainCarRAM-v0-reward-0              # (position, speed) + [reward-1,...,reward-5]
       MountainCarRAM-v0-reward-1              # (position, speed) + reward-1
       MountainCarRAM-v0-reward-2              # (position, speed) + reward-2
       MountainCarRAM-v0-reward-3              # (position, speed) + reward-3
       MountainCarRAM-v0-reward-4              # (position, speed) + reward-4
       MountainCarRAM-v0-reward-5              # (position, speed) + reward-5
       MountainCarImage-v0-reward-0            # image (210, 160, 3) + [reward-1,...,reward-5]
       MountainCarImage-v0-reward-1            # image (210, 160, 3) + reward-1
       MountainCarImage-v0-reward-2            # image (210, 160, 3) + reward-2
       MountainCarImage-v0-reward-3            # image (210, 160, 3) + reward-3
       MountainCarImage-v0-reward-4            # image (210, 160, 3) + reward-4
       MountainCarImage-v0-reward-5            # image (210, 160, 3) + reward-5

     # PS:
       # 自己定义的MountainCar要指定使用RAM还是Image，使用RAM的话是默认返回二元组
         Image是对图像从(400, 600, 3) resize 到 (160, 210, 3) 再转置
       # 定义的Image的目前还不能直接使用baselines训练

  ## Acrobot
     # 新增文件: envs/classic_control/acrobot_reward.py
     # 修改文件: envs/classic_control/__init__.py    # 在此文件导入acrobot_reward.py模块
                 envs/__init__.py                    # 在此文件注册环境名
     # 新增reward类型:
       reward-0 : [reward-1, reward-2]
       reward-1 : 原始的reward
       reward-2 : 高度
     # 最终环境名:
       Acrobot-v1                          # 原始gym提供的reward的环境
       AcrobotRAM-v1-reward-0              # (cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot) + [reward-1, reward-2]
       AcrobotRAM-v1-reward-0              # (cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot) + reward-1
       AcrobotRAM-v1-reward-0              # (cos(t1), sin(t1), cos(t2), sin(t2), t1_dot, t2_dot) + reward-2
       AcrobotImage-v1-reward-0            # image (210, 160, 3) + [reward-1, reward-2]
       AcrobotImage-v1-reward-1            # image (210, 160, 3) + reward-1
       AcrobotImage-v1-reward-2            # image (210, 160, 3) + reward-2

     # PS:
       # 自己定义的Acrobot要指定使用RAM还是Image，使用RAM的话是默认返回六元组，包括两个角度的cos和sin，以及两个角速度
         Image是对图像从(400, 600, 3) resize 到 (160, 210, 3) 再转置
       # 定义的Image的目前还不能直接使用baselines训练
       # 注意Acrobot-v1而不是v0


