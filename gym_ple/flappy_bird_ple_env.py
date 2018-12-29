import os
import gym
from gym import spaces
from ple import PLE
import numpy as np

class PLEFlappyBirdEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, ple_game=True, obs_type="Image", reward_type = 1):
        '''
        For FlappyBird:
            getGameState() returns [player y position, player velocity,
                                    next pipe distance to player, next pipe top y position,
                                    next pipe bottom y position, next next pipe distance,
                                    next next pipe top y, next next pipe bottom y]
        @Params:
            obs_type :
                "RAM" : getGameState()
                "Image" : (512, 288, 3)
            reward_type :
                0 : means [reward1, reward2]
                1 : means raw reward
                2 : means change of y-axis distance from the middle of next top pipe ans bottom pipe
        '''
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        # open up a game state to communicate with emulator
        import importlib
        if ple_game:
            game_module_name = ('ple.games.%s' % game_name).lower()
        else:
            game_module_name = game_name.lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)()

        ##################################################################
        # old one
        #self.game_state = PLE(game, fps=30, display_screen=display_screen)

        # use arg state_preprocessor to support self.game_state.getGameState()
        self.game_state = PLE(game, fps=30, display_screen=display_screen, state_preprocessor = self.process_state)
        ##################################################################

        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.viewer = None

        ##################################################################
        self.obs_type = obs_type
        self.reward_type = reward_type
        ##################################################################

    #############################################
    # Add state processer
    def process_state(self, state):
        return np.array([state.values()])
    #############################################

    def _step(self, a):
        #############################################
        # old observation
        old_ram = self.game_state.getGameState()
        #############################################

        reward = self.game_state.act(self._action_set[a])
        state = self._get_image()
        terminal = self.game_state.game_over()

        #############################################
        # new observation
        ram = self.game_state.getGameState()
        #############################################

        #############################################
        # reward 2
        if self.reward_type == 2:
            reward = self.get_reward(old_ram, ram, terminal, 2)

        # reward 0
        if self.reward_type == 0:
            reward1 = reward
            reward2 = self.get_reward(old_ram, ram, terminal, 2)
            reward = [reward1, reward2]
        ##############################################

        ##############################################
        # obs
        if self.obs_type == "RAM":
            state = self.game_state.getGameState()
        ##############################################

        return state, reward, terminal, {}

    #############################################
    # Add for reward
    #############################################
    def get_reward(self, old_ram, ram, done, reward_type):
        ''' 
        @Params:
            old_ram, ram : numpy.array, [dict_values([x1, x2, ..., x8])]
            reward_type : 2 , change of y-axis distance from the middle line of the next top and bottom pipe
        '''
        old_ram = list(old_ram[0])
        ram = list(ram[0])

        reward = 0.0
        if not done:
            if reward_type == 2:
                old_py, old_top_y, old_bottom_y = old_ram[0], old_ram[3], old_ram[4]
                py, top_y, bottom_y = ram[0], ram[3], ram[4]
                old_dis = abs(old_py - (old_top_y + old_bottom_y) / 2.0)
                dis = abs(py - (top_y + bottom_y) / 2.0)
                reward = 0.1 * (old_dis - dis)
        return reward
    #############################################
    #############################################

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.game_state.reset_game()
        state = self._get_image()
        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()
