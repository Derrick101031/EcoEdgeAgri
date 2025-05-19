# envs/greenhouse_env.py
import gymnasium as gym, numpy as np

class GreenhouseEnv(gym.Env):
    """State = [moist, temp, hum]; Action = {0,1,2}"""
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.action_space  = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.ptr = 0

    def _get_state(self):
        row = self.df.iloc[self.ptr][['field1','field2','field3']].to_numpy()
        # scale back to 0-1 like MCU
        moist = row[0] / 100.0
        temp  = (row[1] + 10) / 60.0
        hum   = row[2] / 100.0
        return np.array([moist,temp,hum], dtype=np.float32)

    def reset(self, **kwargs):
        self.ptr = 0
        return self._get_state(), {}

    def step(self, action:int):
        # very simple: treat logged next row as environment response
        self.ptr += 1
        done = self.ptr >= len(self.df)-1
        next_state = self._get_state()
        # reward = negative absolute error w.r.t. ideal centre-point
        err = np.abs(self.df.loc[self.ptr, 'field1']-30) \
            + np.abs(self.df.loc[self.ptr, 'field2']-22) \
            + np.abs(self.df.loc[self.ptr, 'field3']-70)
        reward = -err
        return next_state, reward, done, False, {}
