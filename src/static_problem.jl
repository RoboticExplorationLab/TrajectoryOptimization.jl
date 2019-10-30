using StaticArrays

"""
$(TYPEDEF)
Store the terms of the 2nd order expansion for the entire trajectory
"""
struct CostExpansion{T,N,M,L1,L2,L3}
    x::Vector{SVector{N,T}}
    u::Vector{SVector{M,T}}
    xx::Vector{SMatrix{N,N,T,L1}}
    uu::Vector{SMatrix{M,M,T,L2}}
    ux::Vector{SMatrix{M,N,T,L3}}
end

function CostExpansion(n,m,N)
    CostExpansion(
        [@SVector zeros(n) for k = 1:N],
        [@SVector zeros(m) for k = 1:N],
        [@SMatrix zeros(n,n) for k = 1:N],
        [@SMatrix zeros(m,m) for k = 1:N],
        [@SMatrix zeros(m,n) for k = 1:N] )
end

function Base.getindex(Q::CostExpansion, k::Int)
    return (x=Q.x[k], u=Q.u[k], xx=Q.xx[k], uu=Q.uu[k], ux=Q.ux[k])
end

"""$(TYPEDEF) Trajectory Optimization Problem.
Contains the full definition of a trajectory optimization problem, including:
* dynamics model (`Model`). Can be either continuous or discrete.
* objective (`Objective`)
* constraints (`Constraints`)
* initial and final states
* Primal variables (state and control trajectories)
* Discretization information: knot points (`N`), time step (`dt`), and total time (`tf`)

# Constructors:
```julia
Problem(model, obj X0, U0; integration, constraints, x0, xf, dt, tf, N)
Problem(model, obj, U0; integration, constraints, x0, xf, dt, tf, N)
Problem(model, obj; integration, constraints, x0, xf, dt, tf, N)
```
# Arguments
* `model`: Dynamics model. Can be either `Discrete` or `Continuous`
* `obj`: Objective
* `X0`: Initial state trajectory. If omitted it will be initialized with NaNs, to be later overwritten by the solver.
* `U0`: Initial control trajectory. If omitted it will be initialized with zeros.
* `x0`: Initial state. Defaults to zeros.
* `xf`: Final state. Defaults to zeros.
* `dt`: Time step
* `tf`: Final time. Set to zero to specify a time penalized problem.
* `N`: Number of knot points. Defaults to 51, unless specified by `dt` and `tf`.
* `integration`: One of the defined integration types to discretize the continuous dynamics model. Defaults to `:none`, which will pass in the continuous dynamics (eg. for DIRCOL)
Both `X0` and `U0` can be either a `Matrix` or a `Vector{Vector}`, but must be the same.
At least 2 of `dt`, `tf`, and `N` need to be specified (or just 1 of `dt` and `tf`).
"""
struct StaticProblem{L<:AbstractModel,T<:AbstractFloat,N,M,NM}
    model::L
    obj::Objective
    constraints::ConstraintSets
    x0::SVector{N,T}
    xf::SVector{N,T}
    Z::Vector{KnotPoint{T,N,M,NM}}
    Z̄::Vector{KnotPoint{T,N,M,NM}}
    N::Int
    dt::T
    tf::T
end

"Get number of states, controls, and knot points"
Base.size(prob::StaticProblem{L,T,N,M,NM}) where {L,T,N,M,NM} = (N, M, prob.N)

"Get the state trajectory"
state(prob::StaticProblem) = [state(z) for z in prob.Z]
"Get the control trajectory"
control(prob::StaticProblem) = [state(prob.Z[k]) for k = 1:prob.N - 1]

TrajectoryOptimization.num_constraints(prob::StaticProblem) = prob.constraints.p
