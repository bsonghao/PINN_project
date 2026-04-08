import torch
import numpy as np

import boundary_conditions

class _Problem:
    "Base problem class to be inherited by different problem classes"

    @property
    def name(self):
        "Defines a name string (only used for labelling automated training runs)"
        raise NotImplementedError

    def __init__(self):
        raise NotImplementedError

    def physics_loss(self, x, *yj):
        "Defines the PINN physics loss to train the NN"
        raise NotImplementedError

    def get_gradients(self, x, y):
        "Returns the gradients yj required for this problem"

    def boundary_condition(self, x, *yj, args):
        "Defines the hard boundary condition to be applied to the NN ansatz"
        raise NotImplementedError


class Burgers2D(_Problem):
    """Solves the time-dependent 1D viscous Burgers equation
        du       du        d^2 u
        -- + u * -- = nu * -----
        dt       dx        dx^2

        for -1.0 < x < +1.0, and 0 < t

        Boundary conditions:
        u(x,0) = - sin(pi*x)
        u(-1,t) = u(+1,t) = 0
    """

    @property
    def name(self):
        return "Burgers2D_nu%.3f"%(self.nu)

    def __init__(self, nu=0.01/np.pi):

        # input params
        self.nu = nu

        # dimensionality of x and y
        self.d = (2,1)

    def physics_loss(self, x, y, j0, j1, jj1):

        physics = (j0[:,0] + y[:,0] * j1[:,0]) - (self.nu * jj1[:,0])# be careful to slice correctly (transposed calculations otherwise (!))
        return torch.mean((physics-0)**2)

    def get_gradients(self, x, y):

        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        jj = torch.autograd.grad(j1, x, torch.ones_like(j1), create_graph=True)[0]
        jj1 = jj[:,1:2]

        return y, j0, j1, jj1

    def boundary_condition(self, x, y, j0, j1, jj1, sd=1):

        # Apply u = tanh((x+1)/sd)*tanh((x-1)/sd)*tanh((t-0)/sd)*NN - sin(pi*x)   ansatz

        t1, jt1, jjt1 = boundary_conditions.tanhtanh_2(x[:,1:2], -1, 1, sd)
        t0, jt0 = boundary_conditions.tanh_1(x[:,0:1], 0, sd)

        sin = -torch.sin(np.pi*x[:,1:2])
        cos = -np.pi*torch.cos(np.pi*x[:,1:2])
        sin2 = (np.pi**2)*torch.sin(np.pi*x[:,1:2])

        y_new   = t0*t1*y                             + sin
        j0_new  = jt0*t1*y + t0*t1*j0
        j1_new  = t0*jt1*y + t0*t1*j1                 + cos
        jj1_new = t0*jjt1*y + 2*t0*jt1*j1 + t0*t1*jj1 + sin2

        return y_new, j0_new, j1_new, jj1_new
