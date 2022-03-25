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
Problem(model, obj, constraints, x0, xf, Z, N, tf)
Problem(model, obj, x0, tf; xf, constraints, N, X0, U0, dt, integration)
```
where `Z` is a trajectory (Vector of `KnotPoint`s)

# Arguments
* `model`: A `DiscreteDynamics` model. If a `ContinuousDynamics` model is provided, it will
           be converted to a `DiscretizedDynamics` model via the integrator specified by the
           `integration` keyword argument.
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
    model::Vector{<:DiscreteDynamics}
    obj::AbstractObjective
    constraints::ConstraintList
    x0::MVector
    xf::MVector
    Z::SampledTrajectory
    N::Int
    t0::T
    tf::T
    function Problem(models::Vector{<:DiscreteDynamics}, obj::AbstractObjective,
            constraints::ConstraintList,
            x0::StaticVector, xf::StaticVector,
            Z::SampledTrajectory, N::Int, t0::T, tf::T) where {Q,T}
        nx,nu = RD.dims(models)
        @assert length(x0) == nx[1]
        @assert length(xf) == nx[end]
        @assert length(Z) == N
        @assert length(models) == N-1 
        @assert tf > t0
        # @assert RobotDynamics.state_dim(obj) == n  "Objective state dimension doesn't match model"
        # @assert RobotDynamics.control_dim(obj) == m "Objective control dimension doesn't match model"
        constraints.nx == nx || throw(DimensionMismatch("Constraint state dimensions don't match model"))
        constraints.nu == nu || throw(DimensionMismatch("Constraint control dimensions don't match model"))
        nx_obj, nu_obj = RD.dims(obj)
        nx_obj == nx || throw(DimensionMismatch("Objective state dimensions don't match model."))
        nu_obj == nu || throw(DimensionMismatch("Objective control dimensions don't match model."))
        # @assert RobotDynamics.dims(Z) == (n,m,N) "Trajectory sizes don't match"
        # TODO: validate trajectory size
        new{T}(models, obj, constraints, x0, xf, Z, N, t0, tf)
    end
end

function Problem(models::Vector{<:DiscreteDynamics}, obj::O, x0::AbstractVector, tf::Real;
        xf::AbstractVector = fill(NaN, state_dim(models[end])),
        constraints=ConstraintList(models),
        t0=zero(tf),
        X0=[fill(NaN, n) for n in RD.dims(models)[1]],
        U0=[fill(0.0, RD.control_dim(model)) for model in models],
        dt=fill((tf-t0)/(length(obj)-1),length(obj)-1)) where {O}

    # Check control dimensions
    nx,nu = RD.dims(models)
    same_state_dimension = all(x->x == nx[1], nx)
    same_control_dimension = all(x->x == nu[1], nu)
    Nx = same_state_dimension ? nx[1] : Any
    Nu = same_control_dimension ? nu[1] : Any
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
    Z = SampledTrajectory{Nx,Nu}(X0,U0,dt=dt)

    Problem(models, obj, constraints, MVector{nx[1]}(x0), MVector{nx[end]}(xf),
        Z, N, t0, tf)
end

function Problem(model::DiscreteDynamics, obj::Objective, args...; kwargs...)
    N = length(obj)
    models = [copy(model) for k = 1:N-1]
    Problem(models, obj, args...; kwargs...)
end

function Problem(model::AbstractModel, args...; 
                 integration::RD.QuadratureRule=RD.RK4(model), kwargs...)
    discrete_model = RD.DiscretizedDynamics(model, integration)
    Problem(discrete_model, args...; kwargs...)
end

"$(TYPEDSIGNATURES)
Get number of states, controls, and knot points"
RD.dims(prob::Problem) = state_dim(prob.model[1]), control_dim(prob.model[1]), prob.N

import Base.size
@deprecate size(prob::Problem) dims(prob) 

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
function initial_trajectory!(prob::Problem, Z0::SampledTrajectory)
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


"Get the number of constraint values at each time step"
num_constraints(prob::Problem) = get_constraints(prob).p
"Get problem constraints. Returns `AbstractConstraintSet`."
@inline get_constraints(prob::Problem) = prob.constraints
"Get the dynamics model. Returns `RobotDynamics.AbstractModel`."
@inline get_model(prob::Problem) = prob.model
"Get the objective. Returns an `AbstractObjective`."
@inline get_objective(prob::Problem) = prob.obj
"Get the trajectory. Returns an `RobotDynamics.SampledTrajectory`"
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

function rollout!(sig::FunctionSignature, models::Vector{<:DiscreteDynamics}, 
                  Z::RD.AbstractTrajectory, x0)
    RD.setstate!(Z[1], x0)
    for k = 2:length(Z)
        RobotDynamics.propagate_dynamics!(sig, models[k-1], Z[k], Z[k-1])
    end
end

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
function add_dynamics_constraints!(prob::Problem, idx=-1; 
        diffmethod=ForwardAD(), sig=StaticReturn())
	n,m = dims(prob)
    conSet = prob.constraints

    # Implicit dynamics
    dyn_con = DynamicsConstraint(prob.model)
    add_constraint!(conSet, dyn_con, 1:prob.N-1, idx, sig=sig, diffmethod=diffmethod) # add it at the end

    # Initial condition
    init_con = GoalConstraint(n, prob.x0, SVector{n}(1:n))  # make sure it's linked
    add_constraint!(conSet, init_con, 1, 1)  # add it at the top

    return nothing
end
