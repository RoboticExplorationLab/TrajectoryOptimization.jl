# 2. Setting up a Cost Function

```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["costfunctions.md"]
```

# Overview
All trajectory optimization problems require a cost function at each stage of the trajectory. Cost functions must be scalar-valued. We assume general cost functions of the form,
```math
\ell_f(x_N) + \sum_{k=1}^{N-1} \ell_k(x_k,u_k)
```
It is very important to note that ``\ell_k(x_k,u_k)`` is ONLY a function of ``x_k`` and ``u_k``, i.e. no coupling across time-steps is permitted. This is a requirement for Differential Dynamic Programming methods such as iLQR, but could be relaxed for methods that parameterize both states and controls, such as DIRCOL (although this is currently not supported). In general, any coupling between adjacent time-steps can be resolved by augmenting the state and defining the appropriate dynamics (this is the method we use to solve minimum time problems).

In general, trajectory optimization will take a second order Taylor series approximation of the cost function, resulting in a quadratic cost function of the form
```math
x_N^T Q_f x_N + q_f^T x_N + \sum_{k=1}^{N-1} x_k^T Q_k x_k + q_k^T x_k + u_k^T R_k u_k + r_k^T u_k + u_k^T H_k x_k
```
This type of quadratic cost is typical for trajectory optimization problems, especially when Q is positive semi-definite and R is positive definite, which is strictly convex. These problem behave well and reduce the computational requirements of taking second-order Taylor series expansions of the cost at each iteration.

# Creating a Cost function
There are several different cost function types that all inherit from `CostFunction`. The following sections detail the various methods for instantiating these cost function types.

## Quadratic Costs
Quadratic costs are the most standard cost function and excellent place to start. Let's assume we are creating an LQR tracking cost of the form
```math
(x_N - x_f)^T Q_f (x_N - x_f) + \sum_{k=1}^{N-1} (x_k - x_f)^T Q (x_k - x_f) + u_k^T R u_k
```
for the simple pendulum with the goal of doing a swing-up. To do this we have a very convenient constructor

```julia
using LinearAlgebra
n,m = 2,1
Q = Diagonal(0.1I,n)
R = Diagonal(0.1I,m)
Qf = Diagonal(1000I,n)
xf = [π,0]
costfun = LQRCost(Q,R,Qf,xf)
```
It is HIGHLY recommended to specify any special structure, such as `Diagonal`, especially since these matrices are almost always diagonal.

This constructor actually does a simple conversion to turn our cost function into a generic quadratic cost function. We could do this ourselves:
```julia
H = zeros(m,n)
q = -Q*xf
r = zeros(m)
c = xf'Q*xf/2
qf = -Qf*xf
cf = xf'Qf*xf/2
costfun = QuadraticCost(Q, R, H, q, r, c, Qf, qf, cf)
```
```@docs
QuadraticCost
```

## Generic Costs
For general, non-linear cost functions use [GenericCost](@ref). Generic cost functions must define their second-order Taylor series expansion, either automatically using `ForwardDiff` or analytically.

Let's say we wanted to use the nonlinear objective for the pendulum
```math
cos(\theta_N) + \omega_N^2 \sum_{k=1}^{N-1} cos(\theta_k) + u_k^T R + u_k + Q ω^2
```
which is small when θ = π, encouraging swing-up.

We define the cost function by defining ℓ(x,u) and ``\ell_f(x)``
```julia
# Define the stage and terminal cost functions
function mycost(x,u)
    R = Diagonal(0.1I,1)
    Q = 0.1
    return cos(x[1] + u'R*u + Q*x[2]^2)
end
function mycost(xN)
    return cos(xN[1]) + xN[2]^2
end

# Create the nonlinear cost function
nlcost = GenericCost(mycost,mycost,n,m)
```
This will use `ForwardDiff` to generate the gradient and Hessian needed for the 2nd order expansion.

Performance-wise, it will be faster to specify the Jacobian analytically (which could also use `ForwardDiff` for part of it). We just need to define the following functions
* `hess`: multiple-dispatch function of the form,
    `Q,R,H = hess(x,u)` with sizes (n,n), (m,m), (m,n)
    `Qf = hess(xN)` with size (n,n)
* `grad`: multiple-dispatch function of the form,
    `q,r = grad(x,u)` with sizes (n,), (m,)
    `qf = grad(x,u)` with size (n,)

Here's an example for the nonlinear cost function we used before
```julia
# Define the gradient and Hessian functions
R = Diagonal(0.1I,m)
Q = 0.1
function hess(x,u)
    n,m = length(x),length(u)
    Qexp = Diagonal([-cos(x[1]), 2Q])
    Rexp = 2R
    H = zeros(m,n)
    return Qexp,Rexp,Hexp
end
function hess(x)
    return Diagonal([-cos(x[1]), 2])
end
function grad(x,u)
    q = [-sin(x[1]), 2Q*x[2]]
    r = 2R*u
    return q,r
end
function grad(x)
    return [-sin(x[1]), 2*x[2]]
end

# Create the cost function
nlcost = GenericCost(mycost, mycost, grad, hess, n, m)
```
