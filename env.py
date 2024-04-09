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
        self.simulation = name
        match self.simulation:
            case "5N-spring2D":
                self.N = 5
                self.D = 2
                self.m = 1

                self.t = 10
                self.dt = 0.05
                
                self.T = 300
                self.ks = 5 
                self.ts = 0
                self.SoB = 5 # size of box
                self.r0 = 1

                self.v = np.zeros((self.N, self.D))
                self.r = np.random.random((self.N, self.D)) * self.SoB

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

                v = np.zeros((self.N, self.D))
                r = np.random.random((self.N, self.D)) * self.SoB

        return np.concatenate(v,r)

    def step(self, forces: npt.ArrayLike ) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, bool]:
        """
        Take a step in the Molecular dynamics simulation 
        Input:
            forces -- the forces acting on the atoms in the system 
        output:
         sim_v, sim_r -- The next state according to the molecular simulation
         v, r -- the next state according to the forces given 
         done -- whether the simulation is finished
        """
        if forces.shape != (self.N, self.D):
            raise ValueError(f"forces must be in shape ({self.N}, {self.D})")

        match self.simulation:
            case "5N-spring2D":
                
                self.ts += 1
                done = False

                if int(self.t / self.dt) >= self.ts:
                    done = True

                f = self.compute_forces()
                sim_v, sim_r = self.euler_int(f)
                self.v, self.r = self.euler_int(forces)

                return (sim_v, sim_r, self.v, self.r, done)
        
    
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
        f = np.zeros((self.N, 2))

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    rij = self.r[i] - self.r[j]
                    rij_abs = np.linalg.norm(rij)
                    f[i] -= self.ks * (rij_abs - self.r0) * rij / rij_abs
        return f

    def euler_int(self, force: list[float]) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Utilizes the euler method to itegrate the velocity and position with the given forces
        """
        v = self.v + force/self.m * self.dt
        r = self.r + self.v * self.dt
        return (v, r)