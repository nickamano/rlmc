import numpy as np
import numpy.typing as npt


class rlmc_env:
    """
    molecular dynamics environment for rienforcement learning
    "5N-spring2D" -- Simulation of 5 atoms connected with Hooks Law with random staring locations and zero velocity
    """

    def __init__(self, name: str, n: int, dt: float, reward_flag:str = "threshold_energy") -> None:
        self.max_int = 65535
        self.seed = np.random.randint(self.max_int)
        np.random.seed(self.seed)
        self.simulation = name

        self.eps_hybrid = 0.05

        match self.simulation:
            case "N-spring2D":
                self.N = n
                self.D = 2
                self.m = 1
                self.reward_flag = reward_flag

                self.dt = dt  # time step

                # Simulation Constants
                self.ks = 1  # Spring Constant
                self.radius = 0.1  # Molecule Radius

                self.ts = 0  # current time step
                self.SoB = 5  # size of box

                self.r_init = np.zeros((self.N, self.D))
                self.v_init = np.zeros((self.N, self.D))
                self.v = self.v_init
                self.r = self.r_init
                self.center = self.r.mean(axis = 0)
                self.v_average = self.v.mean(axis = 0)
                self.terminate = False

                self.U_init = 0
                self.K_init = 0

            case "N-lj2D":
                self.N = n
                self.D = 2
                self.m = 1
                self.reward_flag = reward_flag

                self.dt = dt # time step
                
                # Simulation Constants
                self.radius = 0.005   # Molecule Radius
                self.sig = .45
                self.eps = 7.05081354867767e-02
                self.T = 300
                self.rc = .5 # truncated LJ
                atr = (self.sig/self.rc)**6
                rep = atr*atr
                # print(atr, rep)
                self.A = 48*self.eps/self.rc*(rep-0.5*atr)
                self.B = -4*self.eps*(13*rep-7*atr)

                self.ts = 0 # current time step
                self.SoB = 5.0 # size of box

                self.r_init = np.zeros((self.N, self.D))
                self.v_init = np.zeros((self.N, self.D))
                self.v = self.r_init
                self.r = self.v_init
                self.range = np.linalg.norm(np.max(self.r, axis = 0) - np.min(self.r, axis = 0)) * 1.1
                self.center = self.r.mean(axis = 0)
                self.v_average = self.v.mean(axis = 0)
                self.terminate = False

                self.U_init = 0
                self.K_init = 0
            case _:
                raise NotImplementedError("environment currently not implemented")

    def NNdims(self):
        """
        Return the input and output dimensions of the simulation.
        Use for defining NN input and output sizes
        """
        in_dim = self.N * self.D
        out_dim = self.N * self.D
        return in_dim, out_dim

    def reset(self) -> None:
        """
        Reset the molecular dynamics simulation to initial states
        """
        match self.simulation:
            case "N-spring2D":

                self.v = self.v_init
                self.r = self.r_init
                self.center = self.r.mean(axis = 0)
                self.v_average = self.v.mean(axis = 0)
                self.ts = 0
                self.terminate = False

                self.set_initial_energies()
            case "N-lj2D":
                    self.v = self.v_init
                    self.r = self.r_init
                    self.range = np.linalg.norm(np.max(self.r, axis = 0) - np.min(self.r, axis = 0))
                    self.center = self.r.mean(axis = 0)
                    self.v_average = self.v.mean(axis = 0)
                    self.ts = 0
                    self.terminate = False

    def reset_random(self, max_dist: float) -> None:
        """
        Reset simulation to randomized initial state
        Use when agent reaches acceptable average reward to change initial conditions
        """
        match self.simulation:
            case "N-spring2D":
                self.r_init = max_dist * np.random.rand(self.N, self.D)
                self.v_init = np.random.normal(0,1, (self.N, self.D))

                self.reset()
            case "N-lj2D":
                self.r_init = max_dist * np.random.rand(self.N, self.D)
                self.v_init = np.random.normal(0,1, (self.N, self.D))

                self.reset()

    def set_seed(self, seed: int) -> None:
        """
        Sets the random seed of the enviroment
        """
        self.seed = seed
        np.random.seed(self.seed)

    def set_initial_pos(self, pos: npt.ArrayLike) -> None:
        """
        Sets the initial positions of the environment
        """
        match self.simulation:
            case "N-spring2D":
                if np.array(pos).shape != self.r.shape:
                    raise IndexError("Shape must match shape of system")
                self.r_init = pos
                self.r = self.r_init
                self.center = self.r.mean(axis = 0)
            case "N-lj2D":
                if np.array(pos).shape != self.r.shape:
                    raise IndexError("Shape must match shape of system")
                self.r_init = pos
                self.r = self.r_init
                self.range = np.linalg.norm(np.max(self.r, axis = 0) - np.min(self.r, axis = 0))
                self.center = self.r.mean(axis = 0)

    def set_initial_vel(self, vel: npt.ArrayLike) -> None:
        """
        Sets the initial velocities of the environment
        """
        if np.array(vel).shape != self.v.shape:
            raise IndexError("Shape must match shape of system")
        self.v_init = vel
        self.v = self.v_init
        self.v_average = self.v.mean(axis = 0)

    def set_initial_energies(self):
        """
        Set the initial U and K values for reward calculation
        Call before starting simulation
        """
        self.K_init = self.compute_total_K(self.v)
        self.U_init = self.compute_total_U(self.r)

    def get_current_state(self, n_dt: int) -> npt.ArrayLike:
        """
        Return current state as an flattened array
        """
        return self.r.flatten()

    def step(self, forces: npt.ArrayLike, n_dt: int, step: int, offline: str = "offline", verbose=False) -> tuple[npt.ArrayLike, float, bool]:
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

        # Periodic Condition for LJ
        if self.simulation == "N-lj2D":
            actor_r = actor_r % self.SoB
            r_target = r_target % self.SoB

        # Calculate Reward
        reward = self.reward(v_target,r_target, actor_v, actor_r, forces, target_action)
        sim_reward = self.reward(v_target, r_target, v_target, r_target, target_action, target_action)
        force_diff = np.abs(np.subtract(forces, target_action)).mean()

        # Stop simulation if KE is too large
        actor_KE = self.compute_total_K(actor_v)
        if actor_KE > 2*(self.U_init + self.K_init):
            print("Particles Exploded")
            reward -= 10000
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

        if offline == "offline":
            self.v, self.r = (v_target, r_target)
        elif offline == "online":
            self.v, self.r = (actor_v, actor_r)
        elif offline == "hybrid":
            p = np.random.rand()
            if p < self.eps_hybrid:
                self.v, self.r = (actor_v, actor_r)
            else:
                self.v, self.r = (v_target, r_target)

        return self.r.flatten(), reward, force_diff, done

    def compute_forces(self, r) -> npt.ArrayLike:
        """
        The function computes forces on each particle at time step n
        """
        self.U = 0
        f = np.zeros((self.N, 2))
        match self.simulation:
            case "N-spring2D":
                for i in range(self.N):
                    for j in range(self.N):
                        if i != j:
                            rij = r[i] - r[j]
                            rij_abs = np.linalg.norm(rij)
                            f[i] -= self.ks * (rij_abs - 2 * self.radius) * rij / rij_abs
            case "N-lj2D":
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        rij = r[i] - r[j] % self.SoB
                        rij_abs = np.linalg.norm(rij)

                        feps = 4*self.eps
                        teps = 12*feps

                        atr = (self.radius/rij_abs)**6
                        rep = atr * atr
                       

                        fjk = teps*(rep-0.5*atr)/rij_abs - self.A
                        frtk = fjk/rij_abs*rij

                        f[i] += frtk
                        f[j] -= frtk
        return f

    def euler_int(self, v: npt.ArrayLike, r: npt.ArrayLike, force: npt.ArrayLike, dt: float) -> tuple[
        npt.ArrayLike, npt.ArrayLike]:
        """
        Utilizes the euler method to itegrate the velocity and position with the given forces
        """
        next_v = v + force / self.m * dt
        next_r = r + next_v * dt
        return (next_v, next_r)

    def reward(self, v_target, r_target, v_predict, r_predict, action_actor, action_target):
        """
        Calculates the reward for given v and r, should be calculated after updating self.v and self.r
        """
        match self.reward_flag:
            case "force_only":
                reward = -np.abs(np.subtract(action_actor, action_target)).mean()
            case "initial_energy":
                total_energy_init = self.K_init + self.U_init
                total_energy_pred = self.compute_total_K(v_predict) + self.compute_total_U(r_predict)
                reward = -np.abs(np.subtract(r_target, r_predict)).mean() - np.abs(total_energy_init - total_energy_pred)
            case "sim_comparison":
                total_energy_pred = self.compute_total_K(v_predict) + self.compute_total_U(r_predict)
                total_energy_sim = self.compute_total_K(v_target) + self.compute_total_U(r_target)
                reward = -np.abs(np.subtract(r_target, r_predict)).mean() - np.abs(total_energy_sim - total_energy_pred) - np.abs(np.subtract(action_actor, action_target)).mean()
            case "threshold_energy":
                total_energy_pred = self.compute_total_K(v_predict) + self.compute_total_U(r_predict)
                total_energy_sim = self.compute_total_K(v_target) + self.compute_total_U(r_target)

                if np.abs(total_energy_sim - total_energy_pred) > ((total_energy_sim) * .10):
                    reward = -np.abs(np.subtract(r_target, r_predict)).mean() - np.abs(total_energy_sim - total_energy_pred)
                else:
                    reward = -np.abs(np.subtract(r_target, r_predict)).mean()
            case "threshold_center_of_grav":
                total_energy_pred = self.compute_total_K(v_predict) + self.compute_total_U(r_predict)
                total_energy_sim = self.compute_total_K(v_target) + self.compute_total_U(r_target)
                if np.abs(total_energy_sim - total_energy_pred) > ((total_energy_sim) * .1):
                    reward = - 1 * np.abs(np.subtract(r_target, r_predict)).mean() - np.abs(total_energy_init - total_energy_pred) \
                            - np.abs(np.sum(self.center + self.dt * self.ts * self.v_average - np.mean(r_predict, axis = 0)))
                else:
                    reward = - 1 * np.abs(np.subtract(r_target, r_predict)).mean() \
                    - np.abs(np.sum(self.center + self.dt * self.ts * self.v_average - np.mean(r_predict, axis = 0)))
            case "center_of_grav":
                reward = - np.abs(np.subtract(r_target, r_predict)).mean() \
                         - np.abs(np.sum(self.center + self.dt * self.ts * self.v_average - np.mean(r_predict, axis = 0)))
        return reward

    def compute_total_U(self, r):
        """
        Compute the total potential energy of a system with atoms at r locations
        """
        U = 0
        match self.simulation:
            case "N-spring2D":
                for i in range(self.N):
                    for j in range(i, self.N):
                        if i != j:
                            rij = r[i] - r[j]
                            rij_abs = np.linalg.norm(rij)
                            U += 1/2 * self.ks * rij_abs**2

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
            testenv = rlmc_env("N-spring2D", 50, 0.00001, "threshold_energy")

            # Intialize Starting Positions and Velocities
            testenv.set_initial_pos(3 * np.random.rand(testenv.N, testenv.D))
            testenv.set_initial_vel(np.random.normal(0,1, (testenv.N, testenv.D)))

            # Set Initial Energy
            testenv.set_initial_energies()

            # Section 1: Run simulation for n_steps
            n_steps = 3000
            print("Simulation Start")
            tot_reward = 0
            sum_action = np.zeros((testenv.N, testenv.D))
            print("initial pos: {}".format(testenv.r.flatten()))
            print("initial vel: {}".format(testenv.v.flatten()))
            print(f"initial velo: {testenv.v_average}")
            print(f"intial mean velo: {np.mean(testenv.v, axis = 0)}")
            for i in range(n_steps):
                # print("Step {}".format(i))
                n_dt = 1
                state = testenv.get_current_state(n_dt)
                #action = actornetwork(state)

                action = testenv.compute_forces(testenv.r)  # Replace this action with the action from the actor network
                next_state, reward, done = testenv.step(action, n_dt)

                tot_reward += reward
                sum_action += action

                if i%100 == 0: 
                    print("Step{} reward: {}".format(i, reward))
                    # print(f"\t  center: {testenv.center + testenv.v_average* i * testenv.dt}")
                    # print(f"\t  mean: {np.mean(testenv.r, axis = 0)}")
            print("final pos: {}".format(testenv.r.flatten()))
            print("final vel: {}".format(testenv.v.flatten()))
            print("Reward: {}".format(tot_reward))
            print()

            # Section 2: Step simulation forward by n_steps
            # testenv.reset()
            # print("initial pos: {}".format(testenv.r.flatten()))
            # print("initial vel: {}".format(testenv.v.flatten()))
            # next_state, reward, done = testenv.step(sum_action, n_steps)
            # print("final pos: {}".format(testenv.r.flatten()))
            # print("final vel: {}".format(testenv.v.flatten()))
            # print("Reward: {}".format(reward))

            # # Example of how to get current state
            # state = testenv.get_current_state(n_steps)
            # print("Current state: {}".format(state))

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