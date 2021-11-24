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
struct Problem{T<:AbstractFloat}
    model::DiscreteDynamics
    obj::AbstractObjective
    constraints::ConstraintList
    x0::MVector
    xf::MVector
    Z::Traj
    N::Int
    t0::T
    tf::T
    function Problem(model::DiscreteDynamics, obj::AbstractObjective,
            constraints::ConstraintList,
            x0::StaticVector, xf::StaticVector,
            Z::Traj, N::Int, t0::T, tf::T) where {Q,T}
        n,m = size(model)
        @assert length(x0) == length(xf) == n
        @assert length(Z) == N
        @assert tf > t0
        @assert RobotDynamics.state_dim(obj) == n  "Objective state dimension doesn't match model"
        @assert RobotDynamics.control_dim(obj) == m "Objective control dimension doesn't match model"
        @assert constraints.n == n "Constraint state dimension doesn't match model"
        @assert constraints.m == m "Constraint control dimension doesn't match model"
        @assert RobotDynamics.traj_size(Z) == (n,m,N) "Trajectory sizes don't match"
        new{T}(model, obj, constraints, x0, xf, Z, N, t0, tf)
    end
end

function Problem(model::DiscreteDynamics, obj::O, x0::AbstractVector, tf::Real;
        xf::AbstractVector = fill(NaN, state_dim(model)),
        constraints=ConstraintList(state_dim(model), control_dim(model), length(obj)),
        t0=zero(tf),
        X0=[x0*NaN for k = 1:length(obj)],
        U0=[@SVector zeros(size(model)[2]) for k = 1:length(obj)-1],
        dt=fill((tf-t0)/(length(obj)-1),length(obj)-1)) where {O}
    n,m = dims(model)
    N = length(obj)
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

    Problem(model, obj, constraints, SVector{n}(x0), SVector{n}(xf),
        Z, N, t0, tf)
end

function Problem(model::AbstractModel, args...; 
                 integration::RD.QuadratureRule=RD.RK4(model), kwargs...)
    discrete_model = RD.DiscretizedDynamics(model, integration)
    Problem(discrete_model, args...; kwargs...)
end

function Problem{Q}(model::AbstractModel, args...; kwargs...) where {Q<:QuadratureRule}
    discrete_model = RD.DiscretizedDynamics(model, integration)
    Problem(discrete_model, args...; kwargs...)
end


"$(TYPEDSIGNATURES)
Get number of states, controls, and knot points"
Base.size(prob::Problem) = state_dim(prob.model), control_dim(prob.model), prob.N

"""
    controls(::Problem)

Get the control trajectory
"""
controls(prob::Problem) = controls(prob.Z)
controls(x) = controls(get_trajectory(x))

"""
    states(::Problem)

Get the state trajectory.
"""
states(prob::Problem) = states(prob.Z)
states(x) = states(get_trajectory(x))

"""
	get_times(::Problem)

Get the times for all the knot points in the problem.
"""
@inline RobotDynamics.gettimes(prob::Problem) = gettimes(get_trajectory(prob))


"""
	initial_trajectory!(prob::Problem, Z)

Copy the trajectory
"""
function initial_trajectory!(prob::Problem, Z0::AbstractTrajectory)
	Z = get_trajectory(prob)
    for k = 1:prob.N
        Z[k].z = Z0[k].z
    end
end

"""
	initial_states!(::Problem, X0::Vector{<:AbstractVector})
	initial_states!(::Problem, X0::AbstractMatrix)

Copy the state trajectory
"""
@inline initial_states!(prob, X0) = RobotDynamics.setstates!(get_trajectory(prob), X0)


"""
	set_initial_state!(prob::Problem, x0::AbstractVector)

Set the initial state in `prob` to `x0`
"""
function set_initial_state!(prob::Problem, x0::AbstractVector)
    prob.x0 .= x0
end

"""
    set_initial_time!(prob, t0)

Set the initial time of the optimization problem, shifting the time of all points in the trajectory.
Returns the updated final time.
"""
function set_initial_time!(prob, t0::Real)
    Z = get_trajectory(prob)
    Δt = t0 - Z[1].t
    for k in eachindex(Z)
        Z[k].t += Δt
    end
    return Z[end].t 
end

"""
    set_goal_state!(prob::Problem, xf::AbstractVector; objective=true, constraint=true)

Change the goal state. If the appropriate flags are `true`, it will also modify a 
`GoalConstraint` and the objective, assuming it's an `LQRObjective`.
"""
function set_goal_state!(prob::Problem, xf::AbstractVector; objective=true, constraint=true)
    if objective
        obj = get_objective(prob)
        for k in eachindex(obj.cost)
            set_LQR_goal!(obj[k], xf)
        end
    end
    if constraint
        for con in get_constraints(prob)
            if con isa GoalConstraint
                set_goal_state!(con, xf)
            end
        end
    end
    copyto!(prob.xf, xf)
    return nothing
end

"""
	initial_controls!(::Problem, U0::Vector{<:AbstractVector})
	initial_controls!(::Problem, U0::AbstractMatrx)

Copy the control trajectory
"""
@inline initial_controls!(prob, U0) = RobotDynamics.setcontrols!(get_trajectory(prob), U0)

"""
    cost(::Problem)

Compute the cost for the current trajectory
    """
@inline cost(prob::Problem, Z=prob.Z) = cost(prob.obj, Z)

"Copy the problem"
function Base.copy(prob::Problem)
    Problem(prob.model, copy(prob.obj), copy(prob.constraints), copy(prob.x0), copy(prob.xf),
        copy(prob.Z), prob.N, prob.t0, prob.tf)
end


# function max_violation(prob::Problem, Z::Traj=prob.Z)
#     conSet = get_constraints(prob)
#     evaluate!(conSet, Z)
#     max_violation!(conSet)
#     return maximum(conSet.c_max)
# end

"Get the number of constraint values at each time step"
num_constraints(prob::Problem) = get_constraints(prob).p
"Get problem constraints. Returns `AbstractConstraintSet`."
@inline get_constraints(prob::Problem) = prob.constraints
"Get the dynamics model. Returns `RobotDynamics.AbstractModel`."
@inline get_model(prob::Problem) = prob.model
"Get the objective. Returns an `AbstractObjective`."
@inline get_objective(prob::Problem) = prob.obj
"Get the trajectory. Returns an `RobotDynamics.AbstractTrajectory`"
@inline get_trajectory(prob::Problem) = prob.Z
"Determines if the problem is constrained."
@inline is_constrained(prob) = isempty(get_constraints(prob))
"Get the in initial state. Returns an `AbstractVector`."
@inline get_initial_state(prob::Problem) = prob.x0


"""
    change_integration(prob::Problem, Q<:QuadratureRule)

Change dynamics integration for the problem. Returns a new problem.
"""
change_integration(prob::Problem, ::Type{Q}) where Q<:QuadratureRule =
    Problem{Q}(prob)

function Problem{Q}(p::Problem) where Q
    Problem{Q}(p.model, p.obj, p.constraints, p.x0, p.xf, p.Z, p.N, p.t0, p.tf)
end

"""
	rollout!(::Problem)

Simulate the dynamics forward from the initial condition `x0` using the controls in the
trajectory `Z`.
If a problem is passed in, `Z = prob.Z`, `model = prob.model`, and `x0 = prob.x0`.
"""
@inline rollout!(prob::Problem) = rollout!(StaticReturn(), prob)
@inline rollout!(sig::FunctionSignature, prob::Problem) = 
    rollout!(sig, get_model(prob), get_trajectory(prob), get_initial_state(prob))

function Problem(p::Problem; model=p.model, obj=copy(p.obj), constraints=copy(p.constraints),
    x0=copy(p.x0), xf=copy(p.xf), t0=p.t0, tf=p.tf)
    Problem(model, obj, constraints, x0, xf, copy(p.Z), p.N, t0, tf)
end

"""
    add_dynamics_constraints!(prob::Problem, [integration; idx])

Add dynamics constraints to the constraint set. The integration method `integration` 
defaults to the same integration specified in `prob`, but can be changed. The 
argument `idx` specifies the location of the dynamics constraint in the constraint vector.
If `idx == -1`, it will be added at the end of the `ConstraintList`.
"""
function add_dynamics_constraints!(prob::Problem, idx=-1)
	n,m = size(prob)
    conSet = prob.constraints

    # Implicit dynamics
    dyn_con = DynamicsConstraint(prob.model)
    add_constraint!(conSet, dyn_con, 1:prob.N-1, idx) # add it at the end

    # Initial condition
    init_con = GoalConstraint(n, prob.x0, SVector{n}(1:n))  # make sure it's linked
    add_constraint!(conSet, init_con, 1, 1)  # add it at the top

    return nothing
end
