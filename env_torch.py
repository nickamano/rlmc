import torch


class rlmc_env:
    """
    molecular dynamics environment for rienforcement learning
    "5N-spring2D" -- Simulation of 5 atoms connected with Hooks Law with random staring locations and zero velocity
    """

    def __init__(self, name: str, n: int, dt: float) -> None:
        self.max_int = 65535
        self.seed = torch.randint(high=self.max_int, size=(1,)).item()
        torch.manual_seed(self.seed)
        self.simulation = name

        match self.simulation:
            case "N-spring2D":
                self.N = n
                self.D = 2
                self.m = 1

                self.dt = dt  # time step

                # Simulation Constants
                self.ks = 1  # Spring Constant
                self.radius = 0.1  # Molecule Radius

                self.ts = 0  # current time step
                self.SoB = 5  # size of box


                self.r_init = torch.zeros((self.N, self.D))
                self.v_init = torch.zeros((self.N, self.D))

                self.r = self.r_init
                self.v = self.v_init
                self.terminate = False

                self.U_init = 0
                self.K_init = 0

            case "5N-lj2D":
                raise NotImplementedError("next implementation")
            case _:
                raise NotImplementedError("environment currently not implemented")

    def NNdims(self):
        """
        Return the input and output dimensions of the simulation.
        Use for defining NN input and output sizes
        """
        in_dim = 2 * self.N * self.D + 1
        out_dim = self.N * self.D
        return in_dim, out_dim

    def reset(self) -> None:
        """
        Reset the molecular dynamics simulation to initial states
        """
        match self.simulation:
            case "N-spring2D":
                self.set_seed(self.seed)

                self.v = self.v_init
                self.r = self.r_init
                self.ts = 0
                self.terminate = False

                self.set_initial_energies()

    def reset_random(self, max_dist: float) -> None:
        """
        Reset simulation to randomized initial state
        Use when agent reaches acceptable average reward to change initial conditions
        """
        match self.simulation:
            case "N-spring2D":

                # Set random seed
                seed = torch.randint(high=self.max_int, size=(1,)).item()
                self.set_seed(seed)

                # Initialize r_init with random values scaled by max_dist
                self.r_init = max_dist * torch.rand(self.N, self.D)

                # Initialize v_init with zeros
                self.v_init = torch.zeros(self.N, self.D)


                self.reset()

    def set_seed(self, seed: int) -> None:
        """
        Sets the random seed of the enviroment
        """
        self.seed = seed

        torch.manual_seed(self.seed)

    def set_initial_pos(self, pos: torch.Tensor) -> None:
        """
        Sets the initial positions of the environment
        """
        if pos.shape != self.r.shape:
            raise IndexError("Shape must match shape of system")
        self.r_init = pos
        self.r = self.r_init

    def set_initial_vel(self, vel: torch.Tensor) -> None:
        """
        Sets the initial velocities of the environment
        """
        if vel.shape != self.v.shape:
            raise IndexError("Shape must match shape of system")
        self.v_init = vel
        self.v = self.v_init

    def set_initial_energies(self):
        """
        Set the initial U and K values for reward calculation
        Call before starting simulation
        """
        self.K_init = self.compute_total_K(self.v)
        self.U_init = self.compute_total_U(self.r)

    def get_current_state(self, n_dt: int) -> torch.Tensor:
        """
        Return current state as a flattened tensor
        """
        # Concatenate tensors along a new dimension, then flatten
        concatenated = torch.cat((self.v, self.r), dim=0).flatten()
        # Create a tensor from `self.dt * n_dt` and concatenate it at the end of the flattened tensor
        dt_scaled = torch.tensor([self.dt * n_dt], dtype=concatenated.dtype)
        return torch.cat((concatenated, dt_scaled), dim=0)

    def step(self, forces: torch.Tensor, n_dt: int) -> tuple[torch.Tensor, float, bool]:
        """
        Take a step in the Molecular dynamics simulation
        Input:
            forces -- the forces acting on the atoms in the system
        Output:
            v, r -- (torch.Tensor) The next state according to the forces given
            reward -- (float) Reward given to the actor
            done -- (bool) whether the simulation is finished
        """
        try:
            forces = forces.view(self.N, self.D)
        except:
            raise ValueError(f"forces must be in shape ({self.N}, {self.D})")
        if self.terminate:
            raise ValueError("simulation is terminated")

        match self.simulation:
            case "N-spring2D":
                self.ts += n_dt
                done = False

                # Simulation steps
                v_target = self.v.clone()
                r_target = self.r.clone()
                for _ in range(n_dt):
                    target_action = self.compute_forces(r_target)
                    v_target, r_target = self.euler_int(v_target, r_target, target_action, self.dt)

                # Lazy step
                self.v, self.r = self.euler_int(self.v, self.r, forces, n_dt * self.dt)

                # Calculate Reward
                reward = self.reward(r_target, self.v, self.r)

                # Flatten v and r, append scaled dt, and return
                state = torch.cat((self.v.flatten(), self.r.flatten(), torch.tensor([self.dt * n_dt])))
                return state, reward, done

    def compute_forces(self, r) -> torch.Tensor:
    """
    The function computes forces on each particle at time step n
    """
        self.U = 0
        f = torch.zeros((self.N, 2), dtype=r.dtype, device=r.device)
        match self.simulation:
            case "N-spring2D":
                for i in range(self.N):
                    for j in range(self.N):
                        if i != j:
                            rij = r[i] - r[j]
                            rij_abs = torch.norm(rij)
                            f[i] -= self.ks * (rij_abs - 2 * self.radius) * rij / rij_abs
        return f

    def euler_int(self, v: torch.Tensor, r: torch.Tensor, force: torch.Tensor, dt: float) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Utilizes the Euler method to integrate the velocity and position with the given forces
        """
        next_v = v + (force / self.m) * dt
        next_r = r + v * dt
        return (next_v, next_r)


    def reward(self, r_target, v_predict, r_predict):
        """
        Calculates the reward for given v and r, should be calculated after updating self.v and self.r
        """
        K_predict = self.compute_total_K(v_predict)
        U_predict = self.compute_total_U(r_predict)

        total_energy_init = self.K_init + self.U_init
        total_energy_pred = K_predict + U_predict

        reward = -torch.abs(r_target - r_predict).mean() - torch.abs(
            total_energy_init - total_energy_pred)  # Add short term energy reward
        return reward



    def compute_total_U(self, r):
        """
        Compute the total potential energy of a system with atoms at r locations
        """
        U = 0
        match self.simulation:
            case "N-spring2D":
                for i in range(self.N):
                    for j in range(i + 1, self.N):  # start from i + 1 to avoid unnecessary comparisons
                        if i != j:
                            rij = r[i] - r[j]
                            rij_abs = torch.norm(rij)
                            U += 0.5 * self.ks * (rij_abs ** 2)  # Simplified calculation for potential energy

        return U

    def compute_total_K(self, v):
        """
        Compute the total kinetic energy of the system with atoms with velocity v
        """
        K = 0
        match self.simulation:
            case "N-spring2D":
                for i in range(self.N):
                    K += (self.m / 2) * (v[i] * v[i]).sum()
        return K


if __name__ == "__main__":
    import sys

    runtype = sys.argv[1]

    match runtype:
        case "demo":
            # Initialize Environment for 2D N-body spring simulation
            testenv = rlmc_env("N-spring2D", 5, 0.00005)

            # Intialize Starting Positions and Velocities
            testenv.set_initial_pos(3 * torch.rand(testenv.N, testenv.D))
            testenv.set_initial_vel(torch.zeros((testenv.N, testenv.D)))

            # Set Initial Energy
            testenv.set_initial_energies()

            # Section 1: Run simulation for n_steps
            n_steps = 1000
            print("Simulation Start")
            tot_reward = 0
            sum_action = torch.zeros((testenv.N, testenv.D))
            print("initial pos: {}".format(testenv.r.flatten()))
            print("initial vel: {}".format(testenv.v.flatten()))
            for i in range(n_steps):
                # print("Step {}".format(i))
                n_dt = 1
                state = testenv.get_current_state(n_dt)
                # action = actornetwork(state)

                action = testenv.compute_forces(testenv.r)  # Replace this action with the action from the actor network
                next_state, reward, done = testenv.step(action, n_dt)

                tot_reward += reward
                sum_action += action

                if i % 100 == 0:
                    print("Step{} reward: {}".format(i, reward))
            print("final pos: {}".format(testenv.r.flatten()))
            print("final vel: {}".format(testenv.v.flatten()))
            print("Reward: {}".format(tot_reward))
            print()

            # Section 2: Step simulation forward by n_steps
            testenv.reset()
            print("initial pos: {}".format(testenv.r.flatten()))
            print("initial vel: {}".format(testenv.v.flatten()))
            next_state, reward, done = testenv.step(sum_action, n_steps)
            print("final pos: {}".format(testenv.r.flatten()))
            print("final vel: {}".format(testenv.v.flatten()))
            print("Reward: {}".format(reward))

            # Example of how to get current state
            state = testenv.get_current_state(n_steps)
            print("Current state: {}".format(state))

        case "finddts":
            """Acceptable dt for each N"""
            N_list = [5, 10, 20, 50, 100]
            dt_baselines = [0.005, 0.00005, 0.000005, 0.0000001, 0.00000005]
            dt_dict = dict(zip(N_list, dt_baselines))

            print("(N, dt):")
            for n, dt in zip(N_list, dt_baselines):
                print("({}, {})".format(n, dt))

        case _:
            print("Not a valid case")