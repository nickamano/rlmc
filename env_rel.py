import numpy as np
import numpy.typing as npt
from itertools import permutations
from numpy import matlib as mlib

from env import rlmc_env


class rlmc_env_rel(rlmc_env):
    """
    molecular dynamics environment for rienforcement learning
    "5N-spring2D" -- Simulation of 5 atoms connected with Hooks Law with random staring locations and zero velocity
    """

    def __init__(self, name: str, n: int, dt: float, reward_flag:str = "threshold_energy") -> None:
        super().__init__(name, n, dt, reward_flag)

    def get_relative_state(self, pos) -> npt.ArrayLike:
        """
        Return current state as an flattened array relative to a random particle
        """
        rand_ind = np.random.randint(self.N)
        r_rel = self.r - mlib.repmat(self.r[rand_ind, :], self.N, 1)

        return r_rel.flatten()
    
    def fake_step(self, forces: npt.ArrayLike, n_dt: int, step: int, offline: str = "offline", verbose=False) -> tuple[npt.ArrayLike, float, bool]:
        """
        Take a step in the Molecular dynamics simulation
        Input:
            forces -- the forces acting on the atoms in the system
        output:
         v, r -- (np.array) The next state according to the forces given
         reward -- (float) Reward given to the actor
         done -- (bool) whether the simulation is finished
        """
        try:
            forces = forces.reshape((self.N, self.D))
        except:
            raise ValueError(f"forces must be in shape ({self.N}, {self.D})")
        if self.terminate:
            raise ValueError("simulation is terminated")

        self.ts += n_dt
        done = False

        # Simulation steps
        v_target = np.copy(self.v)
        r_target = np.copy(self.r)
        actor_v = np.copy(self.v)
        actor_r = np.copy(self.r)
        for _ in range(n_dt):
            target_action = self.compute_forces(r_target)
            v_target, r_target = self.euler_int(v_target, r_target, target_action, self.dt)

        # Lazy step
        actor_v, actor_r = self.euler_int(self.v, self.r, forces, n_dt * self.dt)

        # Calculate Reward
        reward = self.reward(v_target,r_target, actor_v, actor_r, forces, target_action)
        sim_reward = self.reward(v_target, r_target, v_target, r_target, target_action, target_action)
        force_diff = np.abs(np.subtract(forces, target_action)).mean()

        # Stop simulation if KE is too large
        actor_KE = self.compute_total_K(actor_v)
        if actor_KE > 2*(self.U_init + self.K_init):
            print("Particles Exploded")
            reward -= 100 * (2000 - step)/2000
            done = True

        if verbose:
            print("R{}, {}".format(reward, sim_reward))
            print("Fa", forces.flatten())
            print("Ft", target_action.flatten())
            print("Va", actor_v.flatten())
            print("Vt", v_target.flatten())
            print("KEa", self.compute_total_K(actor_v))
            print("KEt", self.compute_total_K(v_target))
            print("Ua", self.compute_total_U(actor_r))
            print("Ut", self.compute_total_U(r_target))
            print("UI", self.U_init)
            
            print()

        return self.r.flatten(), reward, force_diff, done