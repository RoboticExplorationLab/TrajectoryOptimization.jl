"""$(TYPEDEF) Trajectory Optimization Problem.
Contains the full definition of a trajectory optimization problem, including:
* dynamics model (`Model`)
* objective (`Objective`)
* constraints (`ConstraintSet`)
* initial and final states
* Primal variables (state and control trajectories)
* Discretization information: knot points (`N`), time step (`dt`), and total time (`tf`)

# Constructors:
```julia
Problem(model, obj, constraints, x0, xf, Z, N, tf) # defaults to RK3 integration
Problem{Q}(model, obj, constraints, x0, xf, Z, N, tf) where Q<:QuadratureRule
Problem(model, obj, xf, tf; x0, constraints, N, X0, U0, dt, integration)
Problem{Q}(prob::Problem)  # change integration
```
where `Z` is a trajectory (Vector of `KnotPoint`s)

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
* `integration`: One of the defined integration types to discretize the continuous dynamics model.
Both `X0` and `U0` can be either a `Matrix` or a `Vector{Vector}`, but must be the same.
At least 2 of `dt`, `tf`, and `N` need to be specified (or just 1 of `dt` and `tf`).
"""
struct Problem{Q<:QuadratureRule,T<:AbstractFloat}
    model::AbstractModel
    obj::AbstractObjective
    constraints::ConstraintList
    x0::MVector
    xf::MVector
    Z::Traj
    N::Int
    t0::T
    tf::T
    function Problem{Q}(model::AbstractModel, obj::AbstractObjective,
            constraints::ConstraintList,
            x0::StaticVector, xf::StaticVector,
            Z::Traj, N::Int, t0::T, tf::T) where {Q,T}
        n,m = size(model)
        @assert length(x0) == length(xf) == n
        @assert length(Z) == N
        @assert tf > t0
        new{Q,T}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    end
end

"Use RK3 as default integration"
Problem(model, obj, constraints, x0, xf, Z, N, t0, tf) =
    Problem{RobotDynamics.RK3}(model, obj, constraints, x0, xf, Z, N, t0, tf)

function Problem(model::L, obj::O, xf::AbstractVector, tf;
        constraints=ConstraintList(size(model)...,length(obj)),
        t0=zero(tf),
        x0=zero(xf), N::Int=length(obj),
        X0=[x0*NaN for k = 1:N],
        U0=[@SVector zeros(size(model)[2]) for k = 1:N-1],
        dt=fill((tf-t0)/(N-1),N-1),
        integration=DEFAULT_Q) where {L,O}
    n,m = size(model)
    if dt isa Real
        dt = fill(dt,N)
    end
	@assert sum(dt[1:N-1]) ≈ tf "Time steps are inconsistent with final time"
    if X0 isa AbstractMatrix
        X0 = [X0[:,k] for k = 1:size(X0,2)]
    end
    if U0 isa AbstractMatrix
        U0 = [U0[:,k] for k = 1:size(U0,2)]
    end
    t = pushfirst!(cumsum(dt), 0)
    Z = Traj(X0,U0,dt,t)

    Problem{integration}(model, obj, constraints, SVector{n}(x0), SVector{n}(xf),
        Z, N, t0, tf)
end



"$(TYPEDSIGNATURES)
Get number of states, controls, and knot points"
Base.size(prob::Problem) = size(prob.model)..., prob.N

"""```julia
integration(::Problem)
integration(::DynamicsConstraint)
```
Get the integration rule"""
integration(prob::Problem{Q}) where Q = Q

"```julia
controls(::Problem)
controls(::AbstractSolver)
controls(::Traj)
```
Get the control trajectory
"
controls(prob::Problem) = controls(prob.Z)

"```julia
states(::Problem)
states(::AbstractSolver)
states(::Traj)
```
Get the state trajectory
"
states(prob::Problem) = states(prob.Z)

"""
	get_times(::Problem)

Get the times for all the knot points in the problem.
"""
@inline get_times(prob::Problem) = get_times(get_trajectory(prob))

"```julia
initial_trajectory!(::Problem, Z)
initial_trajectory!(::AbstractSolver, Z)
```
Copy the trajectory "
function initial_trajectory!(prob::Problem, Z::Traj)
    for k = 1:prob.N
        prob.Z[k].z = Z[k].z
    end
end

"```julia
initial_states!(::Union{Problem,AbstractSolver}, X0::Vector{<:AbstractVector})
initial_states!(::Union{Problem,AbstractSolver}, X0::AbstractMatrix)
```
Copy the state trajectory "
function initial_states!(prob::Problem, X0::Vector{<:AbstractVector})
    set_states!(prob.Z, X0)
end

function initial_states!(prob::Problem, X0::AbstractMatrix)
    X0 = [X0[:,k] for k = 1:size(X0,2)]
    set_states!(prob.Z, X0)
end

"""```julia
set_initial_state!(prob::Problem, x0::AbstractVector)
```
Set the initial state in `prob` to `x0`
"""
function set_initial_state!(prob::Problem, x0::AbstractVector)
    prob.x0 .= x0
end

"```julia
initial_controls!(::Union{Problem,AbstractSolver}, U0::Vector{<:AbstractVector})
initial_controls!(::Union{Problem,AbstractSolver}, U0::AbstractMatrx)
```
Copy the control trajectory "
function initial_controls!(prob::Problem, U0::Vector{<:AbstractVector})
    set_controls!(prob.Z, U0)
end

function initial_controls!(prob::Problem, u0::AbstractVector{<:Real})
    U0 = [copy(u0) for k = 1:prob.N]
    initial_controls!(prob, U0)
end

"```julia
cost(::Problem)
cost(::AbstractSolver)
```
Compute the cost for the current trajectory"
@inline cost(prob::Problem) = cost(prob.obj, prob.Z)

"Copy the problem"
function copy(prob::Problem{Q}) where Q
    Problem{Q}(prob.model, copy(prob.obj), copy(prob.constraints), prob.x0, prob.xf,
        copy(prob.Z), prob.N, prob.t0, prob.tf)
end


function max_violation(prob::Problem, Z::Traj=prob.Z)
    conSet = get_constraints(prob)
    evaluate!(conSet, Z)
    max_violation!(conSet)
    return maximum(conSet.c_max)
end

num_constraints(prob::Problem) = get_constraints(prob).p

@inline get_constraints(prob::Problem) = prob.constraints
@inline get_model(prob::Problem) = prob.model
@inline get_objective(prob::Problem) = prob.obj
@inline get_trajectory(prob::Problem) = prob.Z


"```julia
change_integration(prob::Problem, Q<:QuadratureRule)
```
Change dynamics integration for the problem"
change_integration(prob::Problem, ::Type{Q}) where Q<:QuadratureRule =
    Problem{Q}(prob)

function Problem{Q}(p::Problem) where Q
    Problem{Q}(p.model, p.obj, p.constraints, p.x0, p.xf, p.Z, p.N, p.t0, p.tf)
end

"""
	rollout!(::Problem)
	rollout!(model::AbstractModel, Z::Traj, x0)

Simulate the dynamics forward from the initial condition `x0` using the controls in the
trajectory `Z`.
If a problem is passed in, `Z = prob.Z`, `model = prob.model`, and `x0 = prob.x0`.
"""
@inline rollout!(prob::Problem) = rollout!(prob.model, prob.Z, prob.x0)

function Problem(p::Problem; model=p.model, obj=p.obj, constraints=p.constraints,
    x0=p.x0, xf=p.xf, t0=p.t0, tf=p.tf)
    Problem(model, obj, constraints, x0, xf, p.Z, p.N, t0, tf)
end

"```julia
add_dynamics_constraints!(prob::Problem)
```
Add dynamics constraints to the constraint set"
function add_dynamics_constraints!(prob::Problem{Q}, integration=Q, idx=-1) where Q
	n,m = size(prob)
    conSet = prob.constraints

    # Implicit dynamics
    dyn_con = DynamicsConstraint{integration}(prob.model, prob.N)
    add_constraint!(conSet, dyn_con, 1:prob.N-1, idx) # add it at the end

    # Initial condition
    init_con = GoalConstraint(prob.x0)
    add_constraint!(conSet, init_con, 1, 1)  # add it at the top

    return nothing
end
