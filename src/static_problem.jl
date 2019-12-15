export
    Problem,
    change_integration



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
struct Problem{Q<:QuadratureRule,T<:AbstractFloat}
    model::AbstractModel
    obj::AbstractObjective
    constraints::ConstraintSet{T}
    x0::SVector
    xf::SVector
    Z::Traj
    N::Int
    tf::T
    function Problem{Q}(model::AbstractModel, obj::AbstractObjective,
            constraints::ConstraintSet,
            x0::SVector, xf::SVector,
            Z::Traj, N::Int, tf::T) where {Q,T}
        n,m = size(model)
        @assert length(x0) == length(xf) == n
        @assert length(Z) == N
        new{Q,T}(model, obj, constraints, x0, xf, Z, N, tf)
    end
end

"Use RK3 as default integration"
Problem(model, obj, constraints, x0, xf, Z, N, tf) =
    Problem{RK3}(model, obj, constraints, x0, xf, Z, N, tf)

function Problem(model::L, obj::O, xf::AbstractVector, tf;
        constraints=ConstraintSet(length(obj)),
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

    Problem{integration}(model, obj, constraints, SVector{n}(x0), SVector{n}(xf),
        Z, N, tf)
end



"Get number of states, controls, and knot points"
Base.size(prob::Problem) = size(prob.model)..., prob.N
integration(prob::Problem{Q}) where Q = Q
controls(prob::Problem) = controls(prob.Z)
states(prob::Problem) = states(prob.Z)

function initial_trajectory!(prob::Problem, Z::Traj)
    for k = 1:prob.N
        prob.Z[k].z = Z[k].z
    end
end

function initial_states!(prob::Problem, X0::Vector{<:AbstractVector})
    set_states!(prob.Z, X0)
end

function initial_states!(prob::Problem, X0::AbstractMatrix)
    X0 = [X0[:,k] for k = 1:size(X0,2)]
    set_states!(prob.Z, X0)
end

function initial_controls!(prob::Problem, U0::Vector{<:AbstractVector})
    set_controls!(prob.Z, U0)
end

function initial_controls!(prob::Problem, u0::AbstractVector{<:Real})
    U0 = [copy(u0) for k = 1:prob.N]
    initial_controls!(prob, U0)
end

function cost(prob::Problem)
    cost!(prob.obj, prob.Z)
    return sum( get_J(prob.obj) )
end

function copy(prob::Problem{Q}) where Q
    Problem{Q}(prob.model, copy(prob.obj), copy(prob.constraints), prob.x0, prob.xf,
        copy(prob.Z), prob.N, prob.tf)
end

TrajectoryOptimization.num_constraints(prob::Problem) = get_constraints(prob).p

function max_violation(prob::Problem, Z::Traj=prob.Z)
    conSet = get_constraints(prob)
    evaluate!(conSet, Z)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end

@inline get_constraints(prob::Problem) = prob.constraints


"Change dynamics integration"
change_integration(prob::Problem, ::Type{Q}) where Q<:QuadratureRule =
    Problem{Q}(prob)
function Problem{Q}(p::Problem) where Q
    Problem{Q}(p.model, p.obj, p.constraints, p.x0, p.xf, p.Z, p.N, p.tf)
end

@inline rollout!(prob::Problem) = rollout!(prob.model, prob.Z, prob.x0)
