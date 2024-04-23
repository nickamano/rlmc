import numpy as np
import numpy.typing as npt
from numpy import matlib as mlib

from env import rlmc_env


class rlmc_env_rel(rlmc_env):
    """
    molecular dynamics environment for rienforcement learning
    "5N-spring2D" -- Simulation of 5 atoms connected with Hooks Law with random staring locations and zero velocity
    """

    def __init__(self, name: str, n: int, dt: float, reward_flag:str = "threshold_energy") -> None:
        super().__init__(name, n, dt, reward_flag)

    def get_relative_state(self, n_dt: int) -> npt.ArrayLike:
        """
        Return current state as an flattened array relative to a random particle
        """
        rand_ind = np.random.randint(self.N)
        r_rel = self.r - mlib.repmat(self.r[rand_ind, :], self.N, 1)

        return np.append(r_rel.flatten(), self.dt * n_dt)