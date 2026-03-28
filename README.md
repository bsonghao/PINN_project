# Solving Burger equation from PINN

This repo try to reproduce the results of Burger's equation using PINN from 

Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational physics 378 (2019): 686-707.
[https://www.sciencedirect.com/science/article/pii/S0021999118307125)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

**1. Description of the problem**:

Burger's equation along with Dirichet boundary conditions reads as:
```math
\begin{aligned}
&u_t + uu_x - \nu u_{xx} = 0,\quad x\in[-1, 1], \quad t\in[0,1],\\
&u(0,x) = -\sin(\pi x),\\
&u(t,-1) = u(t, 1) = 0,
\end{aligned}
```
where $u(t,x)$ is a variable of interest that describes the fluid velocity or traffic density
and $\nu=\frac{0.01}{\pi}$ is a parameter known as viscosity.

**2. PINN solution of the problem**:

We introduce neural networks $u(x,t;\theta)$ to approximate the solution of the Burger's equation:
```math
u(x,t;\theta)\approx u(x,t)
```

To solve the problem, we minize the loss function $L(\theta)$:

```math
\begin{aligned}
\theta^\ast = argmin_{\theta} \Big(L(\theta)\Big)
\end{aligned}
```

Where the loss function consist of three parts:
$L(\theta) = L_{int}(\theta) + L_{sb}(\theta) + L_{tb}(\theta)$

- Interior loss $L_{int}(\theta)$ is  given by,
```math
\begin{aligned}
&L_{int}(\theta) = \frac{1}{N_{int}}\sum_{n=1}^{N_{int}}r^2_{int}(y_n;\theta),\\
&r_{int}(y_i,\theta) = u_t(y_n;\theta) + uu_x(y_n;\theta) -\nu u_{xx}(y_n;\theta).
\end{aligned}
```

- Spatial boundary loss $L_{sb}(\theta)$ is given by,
```math
\begin{aligned}
&L_{sb}(\theta) = \frac{1}{N_{sb}}\sum_{i=1}^{N_{sb}}r_{sb}^2(t_n,-1;\theta) + \frac{1}{N_{sb}}\sum_{i=n}^{N_{sb}}r_{sb}^2(t_n,1;\theta),\\
&r_{sb}(t_n, -1;\theta) = u(t_n,-1;\theta)-0,\\
&r_{sb}(t_n, 1;\theta) = u(t_n,1;\theta)-0.
\end{aligned}
``` 
- Temporal boundary loss $L_{tb}(\theta)$ is given by,
```math
\begin{aligned}
&L_{tb}(\theta) = \frac{1}{N_{tb}}\sum_{n=1}^{N_{tb}}r_{tb}^2(x_n;\theta),\\
&r_{tb}(x_n;\theta) = u(0,x_n;\theta) + sin(\pi x_n).
\end{aligned}
```   
       
Both the the training set for interior and boundary loss are obtained quasi Monte-Carlo sampling:

$$
S_{int} =\{y_n\}, \quad 1 \leq n \leq N_{int},\quad y_n = (x,t)_n \in D_T=[0,T]\times[-1,1],
$$

$$
S_{sb, -1} =\{t_n, 0 \}, \quad1 \leq n \leq N_{sb,-1}, t_{n,-1} \in [0,T],
$$

$$
S_{sb, 1} =\{t_n, 0 \}, \quad1 \leq n \leq N_{sb,1}, t_{n,1} \in [0,T],
$$

$$
S_{tb}=\{x_n, -\sin(\pi x_n)\}\quad  1 \leq n \leq N_{tb}, x_n \in [-1,1].
$$

with $\{y_n\}$ obtained from Latin Hypercube Sampling ($N_{int}=10000$), $\{t_{n,-1}\}$, $\{t_{n,1}\}$ and $\{x_n\}$ obtained from random sampling ($N_{sb,-1} + N_{sb, 1} + N_{tb}=100$).


**3. FBPINN solution**

As we can see in the simple PINN solution, there is huge difference with the exact solution at the interface where there are rapid (discontinuous) phase changes. This problem caused by the high freqency bias of PINN which is general feature of neural networks.

To address this issue, we introduce the finite basis physics informed neural networks (FBPINN) approach described in:

Moseley, B., Markham, A., & Nissen-Meyer, T. (2023). Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations: B. Moseley et al. Advances in Computational Mathematics, 49(4), 62.
[https://link.springer.com/article/10.1007/s10444-023-10065-9](https://link.springer.com/article/10.1007/s10444-023-10065-9)

We introduce the FBPINN ansatz to approximate the exact solution:

```math
u(\bf{x};\theta) = \mathit{C}\left(\overline{NN}(\bf{x};\theta)\right),
```
where

```math
\begin{aligned}
&\overline{NN}(X;\theta) = \sum_iw_i(X)\cdot\text{unnorm}\circ NN_i(X;\theta_i) \circ \text{norm}_i(X), \\
&X[1] := x,X[2] := t.
\end{aligned}
```
$\mathit{C}$ is a constraining operator which adds the "hard constraints" boundary condition and can be treated as the additional forward layer of the neural network.

In our specific case:

```math
\mathit{C}\left(\overline{NN}(X;\theta)\right) = -\sin(\pi x) + \tanh(x+1)\tanh(x-1)\overline{NN}(X;\theta)
```
$w(X)$ is the window function that confined the input vector locally with in within the subdomain.

```math
\begin{aligned}
w_i(X) = \prod_{j=1}^{2}\phi\left(\frac{X[j]-a^j_i}{\sigma^j_i}\right)\phi\left(\frac{b^j_i-X[j]}{\sigma^j_i}\right)
\end{aligned}
```
where $\phi(x) = \frac{1}{1+e^{-x}}$, $a^j_i$, $b^j_i$ denote the midpoint of the left and right overlapping regions and $\sigma^j_i$ is a set of parameters defined such that the window function is zero outside the overlap region.

* By taking the "divide and conquer" approach, the FBPINN transformed the global optimization problem into many coupled local optimization problems

* By normalizing each subdomain input, FBPINN approach effectively scales each local problem from a high frequency problem to a lower frequency problem.

**4. Compare with the exact anlytical solution**

The exact analytical solution of Buger's equation is given by [Basdevant et al](https://doi.org/10.1016/0045-7930(86)90036-8):

```math
u(x,t) = \frac{-\int_{-\infty}^{+\infty}\sin\Big(\pi(x-\eta)\Big)f\Big(x-\eta\Big)\exp\Big(\frac{-\eta^2}{4\nu t}\Big)d\eta}{\int_{-\infty}^{+\infty}f\Big(x-\eta\Big)\exp\Big(\frac{-\eta^2}{4\nu t}\Big)d\eta},
```
where
```math
f(y) = \exp\Big(-\frac{\cos(\pi y)}{2\pi\nu}\Big)
```

The numerator and denominator can be obtained using Hermite-Gauss quadrature rule:

```math
\int_{-\infty}^{+\infty}\exp\Big(-x^2\Big)g(x)dx \approx \sum_i^nw_if(x_i),
```
where $x_i$ are the root of the Hermite polynomial: $H_n(x_i)=0,\quad i=1,2,...,n$
and $w_i = \frac{2^{n-1}n!\sqrt{\pi}}{n^2H^2_{n-1}(x_i)}$


# Test results

## PINN result

**Sampling point at the boundary and physics collocation**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/Collocation_points.png" alt="Description" width="500">

**Loss function for training**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/loss_function.png" alt="Description" width="500">

**2D heatmap of the solution**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/full_solution.png" alt="Description" width="500">

**Solution at time crosssection: $t=0.25,0.50,0.75$**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/time_cross_section.png" alt="Description" width="500">

## FBPINN results

**Sampling point at the boundary and physics collocation ($2\times3$ subdomain)**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/FBPINN_collocation_points.png" alt="Description" width="500">

**Plot of window function ($2\times3$ subdomain, 0.4 overlap )**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/plot_domain.png" alt="Description" width="500">

**Loss function for training**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/FBPINN_loss_function.png" alt="Description" width="500">

**2D heatmap of the solution**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/FBPINN_full_solution.png" alt="Description" width="500">

**Solution at time crosssection: $t=0.25,0.50,0.75$**

<img src="https://github.com/bsonghao/PINN_project/blob/FBPINN/results/FBPINN_time_cross_section.png" alt="Description" width="500">

