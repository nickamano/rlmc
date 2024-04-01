import numpy as np

class sim_env:
    def __init__(self, n_particles, maxsteps):
        self.n_p = n_particles
        self.maxsteps = maxsteps

        # parameters of the problem
        self.n_p = 5
        self.m = 1.0 #kg
        self.ks = 5 #N/m
        self.r0 = 1. #m

        # setting a timestep to be 50 ms
        self.dt = 0.05 #s

        # Allocating arrays for 2D problem: first axis - time. second axis - particle's number. third - coordinate
        self.v = np.zeros((self.maxsteps+1, self.n_p, 2))
        self.r = np.zeros((self.maxsteps+1, self.n_p, 2))
        self.f = np.zeros((self.maxsteps+1, self.n_p, 2))

        # initial conditions for 3 particles:
        self.r[0,0] = np.array([0., 2.])
        self.r[0,1] = np.array([2., 0.])
        self.r[0,2] = np.array([-1., 0.])
        self.r[0,3] = np.array([0., -2.])
        self.r[0,4] = np.array([-1., 2.])

        self.current_step = 0

    def compute_forces(self):
        '''The function computes forces on each pearticle at time step n'''
        n = self.current_step
        for i in range(self.n_p):
            for j in range(self.n_p):
                if i != j:
                    rij = self.r[n,i] - self.r[n,j]
                    rij_abs = np.linalg.norm(rij)
                    self.f[n, i] -= self.ks * (rij_abs - self.r0) * rij / rij_abs 

    def integration_method(self):
        n = self.current_step
        self.v[n+1] = self.v[n] + self.f[n]/self.m * self.dt
        self.r[n+1] = self.r[n] + self.v[n+1] * self.dt

    def increment_step(self):
        self.current_step = self.current_step + 1

    def step(self):
        self.compute_forces()
        self.integration_method()
        self.increment_step()