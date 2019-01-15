"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

###########################################
from PIL import Image
###########################################


class MountainCarRewardEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, obs_type = 'RAM', reward_type = 1):
        ''' Initialize
        @Params:
            obs_type : 'RAM' returns an array with shape (2, ), (position, velocity)
            reward_type :
                0 means a list of all type rewards
                1 means raw reward
                2 means distance from destination, smaller is better
                3 means velocity, faster is better
                4 means height, higher is better
                5 means energy,
        '''
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        ###########################################
        self.obs_type = obs_type
        self.reward_type = reward_type

        self.rewards_type_list = [2, 3]
        # every reward type's max-abs value
        self.rewards_ths = [1.0, 0.005, 0.01]

        # change observation space:
        if self.obs_type == "Image":
            self.img_width = 84
            self.img_height = 84
            self.img_shape = (self.img_width, self.img_height, 3)
            self.observation_space = spaces.Box(low = 0, high = 255, shape = self.img_shape, dtype = np.uint8)
        ###########################################

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, gamma = 0.99):
        if isinstance(action,np.ndarray):
            action = action[0]
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state

        ################################################
        # save old observation
        old_position, old_velocity = position, velocity
        ################################################

        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0
        self.state = (position, velocity)

        ################################################
        # init viewer
        if self.viewer is None and self.obs_type == "Image":
            self.init_viewer(visible = False)
        ################################################

        ################################################
        # distance from goal_position
        #if self.reward_type == 2:
            #reward = self.get_reward(position, old_position, velocity, old_velocity, done, 2, gamma)
        # velocity
        if self.reward_type == 2:
            reward = self.get_reward(position, old_position, velocity, old_velocity, done, 3, gamma)
        # height
        if self.reward_type == 3:
            reward = self.get_reward(position, old_position, velocity, old_velocity, done, 4, gamma)
        # energy
        #if self.reward_type == 5:
            #reward = self.get_reward(position, old_position, velocity, old_velocity, done, 5, gamma)

        # reward type 0 : all list of reward
        if self.reward_type == 0:
            rewards = []
            rewards.append(reward)
            for rt in self.rewards_type_list:
                reward_i = self.get_reward(position, old_position, velocity, old_velocity, done, rt, gamma)
                rewards.append(reward_i)
            reward = np.array(rewards)
        ################################################

        ############################################################
        # reward scaling
        if self.reward_type == 0:
            for rt in range(len(reward)):
                reward[rt] = reward[rt] / self.rewards_ths[rt]
        else:
            reward = reward / self.rewards_ths[self.reward_type - 1]
        ############################################################

        ################################################
        # observation is image or ram
        # image is numpy.array with shape (400, 600, 3)
        # ram is numpy.array with shape (2,) as : (position, velocity)
        if self.obs_type == 'Image':
            #obs = self.viewer.get_screen()
            obs = self.viewer.get_array()
            # obs with shape (400, 600, 3)
            # reshape to (160, 210, 3) and swap w-axis and h-axis
            img = Image.fromarray(obs)
            img = img.resize((self.img_width, self.img_height), Image.ANTIALIAS)
            obs = np.array(img).astype(np.uint8)
            #obs = np.swapaxes(obs, 0, 1)
        elif self.obs_type == 'RAM':
            obs = np.array(self.state)
        ################################################

        return obs, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])

        ############################################################
        if self.obs_type == "Image":
            initial_obs = np.zeros(self.img_shape, dtype = np.uint8)
        elif self.obs_type == "RAM":
            initial_obs = np.array(self.state)
        ############################################################

        return initial_obs

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    ###############################################
    # get reward
    ###############################################
    def get_reward(self, position, old_position, velocity, old_velocity, done, reward_type, gamma = 0.99):
        ''' Get reward
        '''
        reward = 0.0
        if not done:
            # distance from goal_position
            #if reward_type == 2:
                #reward = old_position - gamma * position
                #reward = gamma * position - old_position
            # velocity
            if reward_type == 2:
                reward = gamma * velocity - old_velocity
            # height
            if reward_type == 3:
                reward = gamma * self._height(position) - self._height(old_position)
            # energy
            #if reward_type == 4:
                #energy = 0.5 * velocity * velocity + 9.8 * self._height(position)
                #old_energy = 0.5 * old_velocity * old_velocity + 9.8 * self._height(old_position)
                #reward = gamma * energy - old_energy
        return reward

    ###############################################

    ###############################################
    # initial viewer
    ###############################################
    def init_viewer(self, visible = False):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, visible = visible)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))
    #####################################################
    # done
    #####################################################


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
