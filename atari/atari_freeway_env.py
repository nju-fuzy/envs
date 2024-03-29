import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Atari dependencies by running 'pip install gym[atari]'.)".format(e))

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram

from PIL import Image

########################################
#import cv2
import os
import sys

#CHICKEN_IMAGE_PATH = "/home/lxcnju/chicken.jpg"
#if not os.path.exists(CHICKEN_IMAGE_PATH):
#    print("No chicken image found in home directory!....")
#    sys.exit()
########################################


class AtariFreewayEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='freeway', obs_type='image', frameskip=(2, 5), repeat_action_probability=0., reward_type = 2):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game, obs_type, frameskip, repeat_action_probability)
        assert obs_type in ('ram', 'image')

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self.seed()

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

        #########################################
        self.reward_type = reward_type

        # every reward type's max-abs value
        # reward type 1 : max-abs = 1.0
        # reward type 2 : max-abs = 5.0
        self.rewards_ths = [1.0, 5.0]
        #########################################


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)
        return [seed1, seed2]

    def step(self, a, gamma = 0.99):
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])

        ############################################################
        pre_ob = self._get_obs()
        ############################################################

        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        ############################################################
        # done?
        done = self.ale.game_over()

        if self.reward_type == 1:
            reward = reward / self.rewards_ths[0]

        # reward_type = 2 : height change
        if self.reward_type == 2:
            reward = self.get_reward(reward, ob, pre_ob, done, self.reward_type, gamma = gamma)

        if self.reward_type == 0:
            reward1 = reward
            reward2 = self.get_reward(reward, ob, pre_ob, done, 2, gamma = gamma)
            reward = np.array([reward1, reward2])
        ############################################################

        ############################################################
        '''
        # reward scaling
        if self.reward_type == 0:
            for rt in range(len(reward)):
                reward[rt] = reward[rt] / self.rewards_ths[rt]
        else:
            reward = reward / self.rewards_ths[self.reward_type - 1]
        '''
        ############################################################

        return ob, reward, done, {"ale.lives": self.ale.lives()}


    ############################################################
    # add self
    ############################################################
    def get_reward(self, src_reward, ob, pre_ob, done, reward_type, gamma = 0.99):
        ''' Get reward from images!
        @Params:
            ob : observation at current state, numpy.array shape = (210, 160, 3)
            pre_ob : observation from last state
            reward_type : int
        @Return:
            move : height change
        '''
        pre_height = self.get_height(pre_ob)
        height = self.get_height(ob)
        move = (src_reward / self.rewards_ths[0]) + (gamma * height - pre_height) / self.rewards_ths[1]

        if done or abs(move) > 10.0:
            move = src_reward / self.rewards_ths[1]

        return move

    def get_height(self, ob):
        ''' Get height of the chicken
        '''
        '''
        # method 1 : use matchTemplate
        part_ob = ob[:, 43 : 51, :]
        res = cv2.matchTemplate(part_ob, self.chicken_img, cv2.TM_CCOEFF_NORMED).reshape((-1, ))
        height = 210 - np.argmax(res)
        '''
        # method 2 : use yellow color pixel number
        part_ob = ob[:, 43 : 51, :].astype(np.float)
        y_color = np.array([255, 255, 150])
        sub_cols = np.sum(np.abs(part_ob - y_color), axis = 2)
        index_cols = np.sum(sub_cols < 100, axis = 1)

        ss = []
        ss.append(np.sum(index_cols[0 : 8]))
        for i in range(0, 202):
            ss.append(index_cols[i + 8] + ss[-1] - index_cols[i])
        ss = np.array(ss)
        height = 202 - np.argmax(ss)
        return height

    ############################################################
    # add finished
    ############################################################

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self._get_ram()
        elif self._obs_type == 'image':
            img = self._get_image()
        return img

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)

ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
