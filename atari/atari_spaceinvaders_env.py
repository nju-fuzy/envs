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


class AtariSpaceInvadersEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='space_invaders', obs_type='image', frameskip=(2, 5), repeat_action_probability=0., reward_type = 2):
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
        self.rewards_ths = [30.0, 5.0]
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
        pre_lives = self.ale.lives()
        ############################################################

        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        ############################################################
        now_lives = self.ale.lives()
        done = self.ale.game_over()

        if self.reward_type == 2:
            reward = self.get_reward(pre_lives, now_lives, done, 2, gamma = gamma)

        if self.reward_type == 0:
            reward1 = reward
            reward2 = self.get_reward(pre_lives, now_lives, done, 2, gamma = gamma)
            reward = np.array([reward1, reward2])
        ############################################################

        ############################################################
        # reward scaling
        if self.reward_type == 0:
            for rt in range(len(reward)):
                reward[rt] = reward[rt] / self.rewards_ths[rt]
        else:
            reward = reward / self.rewards_ths[self.reward_type - 1]
        ############################################################

        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    #############################################
    # get reward
    #############################################
    def get_reward(self, pre_lives, now_lives, done, reward_type, gamma = 0.99):
        # if live reward get 1.0
        # if loss one life, give -5.0 reward
        reward = 0.0
        if not done:
            if reward_type == 2:
                reward = 1.0
                lost_lives = pre_lives - now_lives
                if lost_lives > 0:
                    reward = -5.0 * lost_lives
                if now_lives == 3 and pre_lives == 0:
                    reward = 0.0
        return reward
    #############################################

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
