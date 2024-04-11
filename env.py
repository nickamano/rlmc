from scipy.stats import maxwell
import numpy as np
import numpy.typing as npt

class rlmc_env:
    """
    molecular dynamics environment for rienforcement learning
    "5N-spring2D" -- Simulation of 5 atoms connected with Hooks Law with random staring locations and zero velocity
    """
    def __init__(self, name: str) -> None:
        self.seed = np.random.randint(0, 1000)
        np.random.seed(self.seed)
        self.simulation = name
        match self.simulation:
            case "5N-spring2D":
                self.N = 5
                self.D = 2
                self.m = 1

                self.t = 15 # total time
                self.dt = 0.05 # change in time
                
                self.T = 300
                self.ks = 5 
                self.ts = 0 # current time step
                self.SoB = 5 # size of box
                self.r0 = 1

                self.v = np.zeros((self.N, self.D))
                self.r = np.random.random((self.N, self.D)) * self.SoB
                self.terminate = False

            case "5N-lj2D":
                raise NotImplementedError("next implementation")
            case _:
                raise NotImplementedError("environment currently not implemented")
    
    def reset(self) -> npt.ArrayLike:
        """
        Reset the molecular dyanmics simulation 
        output:
          state: list[float] - initial velocities and positions of simulation
        """
        match self.simulation:
            case "5N-spring2D":
                np.random.seed(self.seed)

                self.v = np.zeros((self.N, self.D))
                self.r = np.random.random((self.N, self.D)) * self.SoB
                self.ts = 0
                self.terminate = False

        return np.concatenate((self.v, self.r)).flatten()

    def step(self, forces: npt.ArrayLike ) -> tuple[ npt.ArrayLike, float, bool]:
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

        match self.simulation:
            case "5N-spring2D":
                
                self.ts += 1
                done = False

                if self.ts >= int(self.t / self.dt):
                    done = True
                    self.terminated = True
                    return (self.v, self.r, 1, done)

                f = self.compute_forces()
                sim_v, sim_r = self.euler_int(f)
                self.v, self.r = self.euler_int(forces)

                reward = self.reward(sim_v, sim_r)
                if reward == -100:
                    done = True
                    self.terminated = True

                return (np.concatenate((self.v, self.r)).flatten(), reward, done)
        
    
    def set_seed(self, seed: int) -> None:
        """
        Sets the random seed of the enviroment
        """
        self.seed = seed

    def set_initial_pos(self, positions: npt.ArrayLike) -> None:
        """
        Sets the initial positions of the environment
        """
        if np.array(positions).shape != self.r.shape:
            raise IndexError("Shape must match shape of system")
        
    def compute_forces(self) -> npt.ArrayLike:
        """
        The function computes forces on each pearticle at time step n
        """
        f = np.zeros((self.N, 2))

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    rij = self.r[i] - self.r[j]
                    rij_abs = np.linalg.norm(rij)
                    f[i] -= self.ks * (rij_abs - self.r0) * rij / rij_abs
        return f

    def euler_int(self, force: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Utilizes the euler method to itegrate the velocity and position with the given forces
        """
        v = self.v + force/self.m * self.dt
        r = self.r + self.v * self.dt
        return (v, r)

    def reward(self, sim_v, sim_r):
        """
        Calculates the reward for given v and r, should be calculated after updating self.v and self.r
        If the absolute difference in simulation and predicted velocities is at least 100 different from the, kill the run
        and give -100 reward
        """
        sim_energy = (self.compute_total_K(sim_v) - self.compute_total_U(sim_r))
        real_energy = (self.compute_total_K(self.v) - self.compute_total_U(self.r))
        reward = 5 - np.abs(np.subtract(self.r, sim_r)).mean() - np.abs(sim_energy - real_energy)

        if (np.mean(np.abs(self.v - sim_v)) > 100) or reward < 0:
            return -100
        return reward

    def compute_total_U(self, r):
        """
        Compute the total potential energy of a system with atoms at r locations
        """
        match self.simulation:
            case "5N-spring2D":
                U = 0
                for i in range(self.N):
                    for j in range(i, self.N):
                        if i != j:
                            rij = r[i] - r[j]
                            rij_abs = np.linalg.norm(rij)
                            U += self.ks/2 * rij_abs
                return U

    def compute_total_K(self, v):
        """
        Compute the total kinetic energy of the system with atoms with velocity v
        """
        match self.simulation:
            case "5N-spring2D":
                K = 0
                for i in range(self.N):
                    K +=(self.m/2) * (v[i] * v[i]).sum()
                return K