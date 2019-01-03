import os
import gym
from gym import spaces
from ple import PLE
import numpy as np

#################################
from PIL import Image
#################################

class PLECatcherEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='Catcher', display_screen=True, ple_game=True, obs_type="Image", reward_type = 1):
        '''
        For Catcher:
            getGameState() returns [player x position, player velocity, fruits x position, fruits y position]
        @Params:
            obs_type :
                "RAM" : getGameState()
                "Image" : (64, 64, 3)
            reward_type :
                0 : means [reward1, reward2]
                1 : means raw reward
                2 : means change of x-axis distance from fruit
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

        ############################################
        self.obs_type = obs_type
        self.reward_type = reward_type

        # change observation space:
        if self.obs_type == "Image":
            self.img_width = 84
            self.img_height = 84
            self.img_shape = (self.img_width, self.img_height, 3)
            self.observation_space = spaces.Box(low = 0, high = 255, shape = self.img_shape, dtype = np.uint8)
        elif self.obs_type == "RAM":
            self.observation_space = spaces.Box(low = -100.0, high = 100.0, shape = (4, ), dtype = np.float32)
        ############################################


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

        #############################################
        #state = self._get_image()
        if self.obs_type == "Image":
            state = self._get_image()
        #############################################

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
            reward = np.array([reward1, reward2])
        ##############################################

        ##############################################
        # obs
        if self.obs_type == "RAM":
            state = self.game_state.getGameState()
            state = np.array(list(state[0]))
        ##############################################

        return state, reward, terminal, {}

    #############################################
    # Add for reward
    #############################################
    def get_reward(self, old_ram, ram, done, reward_type):
        ''' 
        @Params:
            old_ram, ram : numpy.array, [dict_values([x, y, z, w])]
            reward_type : 2 , distance of x-axis change
        '''
        old_ram = list(old_ram[0])
        ram = list(ram[0])
        reward = 0.0
        if not done:
            if reward_type == 2:
                old_px, old_fx = old_ram[0], old_ram[2]
                px, fx = ram[0], ram[2]
                old_dis = abs(old_px - old_fx)
                dis = abs(px - fx)
                reward = 0.1 * (old_dis - dis)
        return reward
    #############################################
    #############################################

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        ##########################################
        # resize image
        img = Image.fromarray(image_rotated)
        img = img.resize((self.img_width, self.img_height), Image.ANTIALIAS)
        image_resized = np.array(img).astype(np.uint8)
        ##########################################
        return image_resized

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.game_state.reset_game()
        #######################################
        if self.obs_type == "Image":
            state = self._get_image()
        elif self.obs_type == "RAM":
            state = self.game_state.getGameState()
            state = np.array(list(state[0]))
        #######################################
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
