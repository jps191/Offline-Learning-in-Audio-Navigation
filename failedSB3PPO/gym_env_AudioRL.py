import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple

class UnityGymWrapper(gym.Env):
    def __init__(self, env_path, worker_id=0, no_graphics=True):
        super().__init__()

        self.env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            no_graphics=no_graphics,
            seed=1
        )

        self.env.reset()

        # Get the first agent behavior
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        spec = self.env.behavior_specs[self.behavior_name]

        # Setup observation space
        obs_dim = spec.observation_specs[0].shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Setup action space (continuous only)
        act_dim = spec.action_spec.continuous_size
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        obs = decision_steps.obs[0][0]
        return obs, {}


    def step(self, action):
        # Convert action → ML-Agents structure
        action_tuple = ActionTuple(continuous=action.reshape(1, -1))
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            # Terminal state
            obs = terminal_steps.obs[0][0]
            reward = float(terminal_steps.reward[0])
            done = True
        else:
            # Ongoing state
            obs = decision_steps.obs[0][0]
            reward = float(decision_steps.reward[0])
            done = False

        return obs, reward, done, False, {}

    def close(self):
        self.env.close()
