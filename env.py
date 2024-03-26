from scipy.stats import maxwell
import numpy as np
import numpy.typing as npt

class rlmc_env:
    """
    molecular dynamics environment for rienforcement learning
    """

    def __init__(self, name: str) -> None:
        self.seed = np.random.randint(0, 1000)
        self.simulation = name
        match self.simulation:
            case "5N-spring2D":
                self.N = 5
                self.m = 1

                self.t = 10
                self.dt = 0.05
                
                self.T = 300
                self.ks = 5 
                self.ts = 0
                self.SoB = 5 # size of box
                self.r0 = 1

                self.v = np.zeros(self.N, 2)
                self.r = np.random.rand(self.N, 2) * self.SoB

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

                v = np.zeros(self.N, 2)
                r = np.random.rand(self.N, 2) * self.SoB

        return np.concatenate(v,r)

    def step(self, forces: npt.ArrayLike ) -> tuple[npt.ArrayLike, npt.ArrayLike, bool]:
        """
        Take a step in the Molecular dynamics simulation 
        take in the forces and 
        return the next state and if the simulation is finished
        """
        self.ts += 1
        done = False

        if int(self.t / self.dt) >= self.ts:
            done = True

        f = self.compute_forces()
        sim_v, sim_r = self.euler_int(f)
        input_v, input_r = self.euler_int(forces)

        return (np.concatenate(sim_v, sim_r), np.concatenate(input_v, input_r), done)
        
    
    def seed(self, seed: int) -> None:
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
        f = np.zeros(self.N, 2)

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    rij = self.r[i] - self.r[j]
                    rij_abs = np.linalg.norm(rij)
                    f[i] -= self.ks * (rij_abs - self.r0) * rij / rij_abs
        return f

    def euler_int(self, force: list[float]) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        v = self.v + force/self.m * self.dt
        r = self.r + self.v * self.dt
        return (v, r)