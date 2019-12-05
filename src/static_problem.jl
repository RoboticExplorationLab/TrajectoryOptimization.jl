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
struct StaticProblem{Q<:QuadratureRule,T<:AbstractFloat,L<:AbstractModel,O<:AbstractObjective,N,M,NM}
    model::L
    obj::O
    constraints::ConstraintSets
    x0::SVector{N,T}
    xf::SVector{N,T}
    Z::Vector{KnotPoint{T,N,M,NM}}
    Z̄::Vector{KnotPoint{T,N,M,NM}}
    N::Int
    tf::T
    function StaticProblem{Q}(model::L, obj::O, constraints::ConstraintSets,
            x0::SVector, xf::SVector,
            Z::Vector{KnotPoint{T,n,m,nm}}, Z̄::Traj, N::Int, tf::T) where {T,Q,L,O,n,m,nm}
        @assert length(Z) == N
        @assert length(Z̄) == N
        @assert length(obj) == N
        @assert traj_size(Z) == (n,m,N)
        @assert traj_size(Z̄) == (n,m,N)
        @assert length(x0) == n
        @assert length(xf) == n
        new{Q,T,L,O,n,m,nm}(model, obj, constraints, x0, xf, Z, Z̄, N, tf)
    end
end

"Use RK3 as default integration"
StaticProblem(model, obj, constraints, x0, xf, Z, Z̄, N, tf) =
    StaticProblem{RK3}(model, obj, constraints, x0, xf, Z, Z̄, N, tf)

function StaticProblem(model::L, obj::O, xf::AbstractVector;
        constraints=ConstraintSets(length(obj)),
        x0=zero(xf), N::Int=length(obj), tf=NaN,
        Z=NaN,
        Z̄=NaN,
        integration=RK3) where {L,O}
    n,m = size(model)
    if isnan(tf)
        if isnan(Z)
            throw(ArgumentError("final time not specified. Must either specify it directly or pass in a trajectory"))
        else
            tf = mapreduce(z->z.dt, +, Z)
        end
    end
    if isnan(Z)
        dt = tf / (N-1)
        Z = Traj(n,m,dt,N)
    end
    if isnan(Z̄)
        dt = tf / (N-1)
        Z̄ = Traj(n,m,dt,N)
    end

    StaticProblem{integration}(model, obj, constraints, SVector{n}(x0), SVector{n}(xf),
        Z, Z̄, N, tf)
end


"Get number of states, controls, and knot points"
Base.size(prob::StaticProblem{Q,T,L,O,N,M,NM}) where {T,Q,L,O,N,M,NM} = (N, M, prob.N)

"Get the state trajectory"
state(prob::StaticProblem) = [state(z) for z in prob.Z]
"Get the control trajectory"
control(prob::StaticProblem) = [control(prob.Z[k]) for k = 1:prob.N - 1]

function initial_trajectory!(prob::StaticProblem, Z::Traj)
    for k = 1:prob.N
        prob.Z[k].z = Z[k].z
    end
end

function initial_controls!(prob::StaticProblem, U0::Vector{<:AbstractVector})
    N = prob.N
    Z = prob.Z
    if length(U0) == N-1
        push!(U0,U0[end])
    end
    for k in 1:N
        x = state(Z[k])
        Z[k].z = [x; U0[k]]
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

function copy(prob::StaticProblem)
    StaticProblem(prob.model, copy(prob.obj), ConstraintSets(copy(prob.constraints.constraints), prob.N), prob.x0, prob.xf,
        deepcopy(prob.Z), deepcopy(prob.Z̄), prob.N, prob.tf)
end

TrajectoryOptimization.num_constraints(prob::StaticProblem) = get_constraints(prob).p

function max_violation(prob::StaticProblem)
    conSet = get_constraints(prob)
    evaluate!(conSet, prob.Z)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end

function change_integration(prob::StaticProblem, ::Type{Q}) where Q<:QuadratureRule
    StaticProblem{Q}(prob.model, prob.obj, prob.constraints, prob.x0, prob.xf,
        prob.Z, prob.Z̄, prob.N, prob.tf)
end
