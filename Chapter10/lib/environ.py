import typing as tt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec
import enum
import numpy as np
from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

# the environment class
class StocksEnv(gym.Env):
    spec = EnvSpec("StocksEnv-v0")
# spec is required for gym.Env compatibility and registers our environment in the Gym internal registry

    def __init__(self, prices: tt.Dict[str, data.Prices],
                    bars_count: int = DEFAULT_BARS_COUNT,
                    commission: float = DEFAULT_COMMISSION_PERC,
                    reset_on_close: bool = True, state_1d: bool = False,
                    random_ofs_on_reset: bool = True,
                    reward_on_close: bool = False, volumes=False):
        self._prices = prices   # contains one or more stocks prices for one or more instructments as a dict
        # where keys are the instrutment's name and the value is a container object data

        # state object
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_closer=reward_on_close, volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close,
            reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = spaces.Discrete(n=len(Actions)) # action space
        self.observation_space = spaces.Box(    # observation space
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32
        )
        self.random_ofs_on_reset = random_ofs_on_reset
    

    def reset(self, *, seed, options):
        # make selection of the instrument and it's offset. Then reset the state
        super().reset(seed=seed, options=options)
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        # restart on random offset of days in that year
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0] - bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode(), {}
        
    # handle the action chosen by the agent and return the next observation, reward and done flag
    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "instrument" : self._instrument,
            "offset" : self._state._offset
        }
        return obs, reward, done, False, info

    # this is a way to create environment with data directory as the argument
    # load all the quotes from the CSV files in the directory and construct the environment
    @classmethod
    def from_dir(cls, data_dir:str, **kwargs):
        prices = {
            file: data.load_relateive(file)
            for file in data.price_files(data_dir)
        }

        return StocksEnv(prices, **kwargs)

class State:
    def __init__(self, bars_count, commission_perc, 
                reset_on_close, reward_on_close, volumes):
        assert bars_count > 0
        assert commission_perc >= 0.0
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.have_position = False
        self.open_price = 0.0
        self._prices = None
        self._offset = None

    def reset(self, prices, offset):
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    # calculate the dimensions of the observation/state vector that will feed into the neural network
    # bar_count - the number of timestamps of the market
    @property
    def shape(self): # position flag = if the current position is holding the stock
        # [h, l, c, v] * bars + position_flag + rel_profit
        if self.volumes:
            return 4 * self.bars_count + 1 + 1
        # [h, l, c] * bars + position_flag + rel_profit
        else:
            return 3 * self.bars_count + 1 + 1
    
    # packs prices at the current offset into NumPy array, which will be the observation of the agent
    def encode(self):
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            ofs = self._offset + bar_idx
            # give in the high, low, close , and volume (if present)
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        # position_flag
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:   # percentage of the current closing price compared to the open_price
            res[shift] = self._cur_close() / self.open_price - 1.0
        return res

    # calculate the current closing price using the relative closing price on top of the openning price
    def _cur_close(self):
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)
    
    def step(self, action):
        reward = 0.0
        done = False
        close = self._cur_close()

        # if the action is buy, and currently does not hold a share
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc # get commission deducated from the reward
        elif action == Actions.Close and self.have_position: # if we currenty hold the share and the action is to sell
            reward -= self.commission_perc  # pay the commission fee
            done |= self.reset_on_close # see if we reset 
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
                self.have_position = False
                self.open_price = 0.0
        self._offset += 1 # change the day - offset
        prev_close = close 
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0] - 1 # check if reached the end of the calendar year, forcefully close

        # if reward is enabled for each step
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close/prev_close - 1.0) # calculate reward for each step
        
        return reward, done

class State1D(State):
    @property
    def shape(self): # high, low, close, volumes, position_flag, rel_profit
        if self.volumes:
            return 6, self.bars_count
        else:   # high, low, close, position flag, relative profit
            return 5, self.bars_count
    
    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        start = self._offset - (self.bars_count - 1)
        stop = self._offset + 1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0  # position flag
            res[dst+1] = self._cur_close() / self.open_price - 1.0 # relateive profit
        return res

    