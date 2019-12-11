export
    StaticProblem


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
struct StaticProblem{Q<:QuadratureRule,T<:AbstractFloat}
    model::AbstractModel
    obj::AbstractObjective
    constraints::ConstraintSets{T}
    x0::SVector
    xf::SVector
    Z::Traj
    N::Int
    tf::T
    function StaticProblem{Q}(model::AbstractModel, obj::AbstractObjective,
            constraints::ConstraintSets,
            x0::SVector, xf::SVector,
            Z::Traj, N::Int, tf::T) where {Q,T}
        n,m = size(model)
        @assert length(x0) == length(xf) == n
        @assert length(Z) == N
        new{Q,T}(model, obj, constraints, x0, xf, Z, N, tf)
    end
end

"Use RK3 as default integration"
StaticProblem(model, obj, constraints, x0, xf, Z, N, tf) =
    StaticProblem{RK3}(model, obj, constraints, x0, xf, Z, N, tf)

function StaticProblem(model::L, obj::O, xf::AbstractVector, tf;
        constraints=ConstraintSets(length(obj)),
        x0=zero(xf), N::Int=length(obj),
        X0=[x0*NaN for k = 1:N],
        U0=[@SVector zeros(size(model)[2]) for k = 1:N-1],
        dt=fill(tf/(N-1),N),
        integration=RK3) where {L,O}
    n,m = size(model)
    if X0 isa AbstractMatrix
        X0 = [X0[:,k] for k = 1:size(X0,2)]
    end
    if U0 isa AbstractMatrix
        U0 = [U0[:,k] for k = 1:size(U0,2)]
    end
    Z = Traj(X0,U0,dt)

    StaticProblem{integration}(model, obj, constraints, SVector{n}(x0), SVector{n}(xf),
        Z, N, tf)
end



"Get number of states, controls, and knot points"
Base.size(prob::StaticProblem) = size(prob.model)..., prob.N
integration(prob::StaticProblem{Q}) where Q = Q
controls(prob::StaticProblem) = controls(prob.Z)
states(prob::StaticProblem) = states(prob.Z)

function initial_trajectory!(prob::StaticProblem, Z::Traj)
    for k = 1:prob.N
        prob.Z[k].z = Z[k].z
    end
end

function initial_controls!(prob::StaticProblem, U0::Vector{<:AbstractVector})
    N = prob.N
    for k in 1:N-1
        prob.U0[k] = U0[k]
    end
end

function initial_controls!(prob::StaticProblem, u0::AbstractVector{<:Real})
    U0 = [copy(u0) for k = 1:prob.N]
    initial_controls!(prob, U0)
end

function cost(prob::StaticProblem)
    cost!(prob.obj, prob.Z)
    return sum( get_J(prob.obj) )
end

function copy(prob::StaticProblem{Q}) where Q
    StaticProblem{Q}(prob.model, copy(prob.obj), copy(prob.constraints), prob.x0, prob.xf,
        copy(prob.Z), prob.N, prob.tf)
end

TrajectoryOptimization.num_constraints(prob::StaticProblem) = get_constraints(prob).p

function max_violation(prob::StaticProblem)
    conSet = get_constraints(prob)
    evaluate!(conSet, prob.Z)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end

@inline get_constraints(prob::StaticProblem) = prob.constraints


"Change dynamics integration"
change_integration(prob::StaticProblem, ::Type{Q}) where Q<:QuadratureRule =
    StaticProblem{Q}(prob)
function StaticProblem{Q}(p::StaticProblem) where Q
    StaticProblem{Q}(p.model, p.obj, p.constraints, p.x0, p.xf, p.Z, p.N, p.tf)
end

@inline rollout!(prob::StaticProblem) = rollout!(prob.model, prob.Z, prob.x0)
