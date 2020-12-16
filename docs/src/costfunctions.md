# [2. Setting up an Objective](@id objective_section)

```@meta
CurrentModule = TrajectoryOptimization
```

```@contents
Pages = ["costfunctions.md"]
```

## Overview
All trajectory optimization problems require a cost function at each stage of the trajectory. Cost functions must be scalar-valued. We assume general cost functions of the form,
```math
\ell_f(x_N) + \sum_{k=1}^{N-1} \ell_k(x_k,u_k) dt
```
It is very important to note that ``\ell_k(x_k,u_k)`` is ONLY a function of ``x_k`` and ``u_k``, i.e. no coupling across time-steps is permitted. This is a requirement for Differential Dynamic Programming methods such as iLQR, but could be relaxed for methods that parameterize both states and controls, such as DIRCOL. In general, any coupling between adjacent time-steps can be resolved by augmenting the state and defining the appropriate dynamics (this is the method we use to solve minimum time problems).

In general, trajectory optimization will take a second order Taylor series approximation of the cost function, resulting in a quadratic cost function of the form
```math
x_N^T Q_f x_N + q_f^T x_N + \sum_{k=1}^{N-1} x_k^T Q_k x_k + q_k^T x_k + u_k^T R_k u_k + r_k^T u_k + u_k^T H_k x_k
```
This type of quadratic cost is typical for trajectory optimization problems, especially when
Q is positive semi-definite and R is positive definite, which is strictly convex.
These problems behave well and reduce the computational requirements of taking second-order
Taylor series expansions of the cost at each iteration.

In TrajectoryOptimization.jl we differentiate between the entire objective and the cost functions at each time step. We use `Objective` to describe the function that is being minimized, which typically consists of a sum of cost functions, with potentially some additional terms (as is the case with augmented Lagrangian objectives). Describing the Objective as a sum of individual functions allows the solvers to more efficiently compute the gradient and Hessian of the entire cost, which is block-diagonal given the Markovianity of the problem.

## Cost functions
While TrajectoryOptimization allows for general nonlinear cost function in principle, currently
only quadratic cost functions are implemented (implementing nonlinear cost functions is a
great way to contribute!). All cost functions inherit from the general `CostFunction` type.

Since quadratic costs are the most standard cost function they excellent place to start.
Let's assume we are creating an LQR tracking cost of the form
```math
(x_N - x_f)^T Q_f (x_N - x_f) + \sum_{k=1}^{N-1} (x_k - x_f)^T Q (x_k - x_f) + u_k^T R u_k
```
for the simple cartpole with the goal of doing a swing-up. To do this we have very convenient
method [`LQRCost`](@ref).
```julia
using LinearAlgebra, StaticArrays
n,m = 4,1
Q = Diagonal(@SVector fill(0.1,n))
R = Diagonal(@SVector fill(0.1,m))
Qf = Diagonal(@SVector fill(1000,n))
xf = @SVector [0,π,0,0]
costfun = LQRCost(Q,R,xf)
costfun_term = LQRCost(Qf,R*0,xf,terminal=true)
```
!!! tip
    It is HIGHLY recommended to specify any special structure, such as `Diagonal`, especially since these matrices are almost always diagonal. See Julia's built-in
    `LinearAlgebra` module for more specialized matrix types.

This constructor actually does a simple conversion to turn our cost function into either the
generic [`QuadraticCost`](@ref) or a [`DiagonalCost`](@ref). We could do this ourselves:
```julia
H = @SMatrix zeros(m,n)
q = -Q*xf
r = @SVector zeros(m)
c = xf'Q*xf/2
qf = -Qf*xf
cf = xf'Qf*xf/2
costfun      = QuadraticCost(Q, R, H, q, r, c)
costfun_term = QuadraticCost(Qf, R*0, H, qf, r*0, cf)
```
The `QuadraticCost` constructor also supports keyword arguments and one that allows for only `Q,q` and `c`.:
```julia
costfun = QuadraticCost(Q, R, q=q, c=c)
```

## Objective
Once we have defined the cost function, we can create an objective for our problem by simply
copying over all time steps (except for the terminal).
```julia
# Create an objective from a single cost function
N = 51
obj = Objective(costfun, costfun_term, N)
```

There's also a convenient constructor that skips all the previous steps and builds
the objective directly, see[`LQRObjective`](@ref)
```julia
obj = LQRObjective(Q, R, Qf, xf, N)
```
