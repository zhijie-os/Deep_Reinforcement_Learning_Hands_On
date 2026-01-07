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

class StocksEnv(gym.Env):
    spec = EnvSpec("StocksEnv-v0")

@classmethod
def from_dir(cls, data_dir:str, **kwargs):
    prices = {
        file: data.load_relateive(file)
        for file in data.price_files(data_dir)
    }

    return StocksEnv(prices, **kwargs)


class StocksEnv(gym.Env):
    spec = EnvSpec("StocksEnv-v0")

    def __init__(self, prices: tt.Dict[str, data.Prices],
                    bars_count: int = DEFAULT_BARS_COUNT,
                    commission: float = DEFAULT_COMMISSION_PERC,
                    reset_on_close: bool = True, state_1d: bool = False,
                    random_ofs_on_reset: bool = True,
                    reward_on_close: bool = False, volumes=False):
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_closer=reward_on_close, volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close,
            reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32
        )
        self.random_ofs_on_reset = random_ofs_on_reset
    
