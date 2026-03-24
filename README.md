# Solving Burger equation from PINN

This repo try to reproduce the results of Burger's equation using PINN from 

Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational physics 378 (2019): 686-707.
[https://www.sciencedirect.com/science/article/pii/S0021999118307125)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

**1. Description of the problem**:

Burger's equation along with Dirichet boundary conditions reads as

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
$$
u(x,t;\theta)\approx u(x,t)
$$

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
&t_{tb}(x_n;\theta) = u(0,x_n;\theta) + sin(\pi x_n).
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

$$
L_{int}(\theta) = \frac{1}{N_{int}}\sum_{i=1}^{N_{int}}r_{int,\theta}^2(y_n), \quad
L_{sb}(\theta) = \frac{1}{N_{sb}}\sum_{i=1}^{N_{sb}}r_{sb,\theta}^2(t_n,-1) + \frac{1}{N_{sb}}\sum_{i=1}^{N_{sb}}r_{sb,\theta}^2(t_n,1), \quad
L_{tb}(\theta) = \frac{1}{N_{tb}}\sum_{i=1}^{N_{tb}}r_{tb,\theta}^2(x_n)
$$


**3. Compare with the exact anlytical solution**

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

## Sampling point at the boundary and physics collocation
<img src="https://github.com/bsonghao/PINN_project/blob/main/results/Collocation_points.png" alt="Description" width="500">


## 2D heatmap of the solution
<img src="https://github.com/bsonghao/PINN_project/blob/main/results/full_solution.png" alt="Description" width="500">

## Solution at time crosssection: $t=0.25,0.50,0.75$
<img src="https://github.com/bsonghao/PINN_project/blob/main/results/time_cross_section.png" alt="Description" width="500">

