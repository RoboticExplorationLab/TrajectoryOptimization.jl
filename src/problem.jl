"""
    Problem{T}

Trajectory Optimization Problem.
Contains the full definition of a trajectory optimization problem, including:
* dynamics model (`RD.DiscreteDynamics`)
* objective ([`Objective`](@ref))
* constraints ([`ConstraintList`](@ref))
* initial and final states
* Primal variables (state and control trajectories)
* Discretization information: knot points (`N`), time step (`dt`), and total time (`tf`)

# Constructors:
```julia
Problem(model, obj, constraints, x0, xf, Z, N, tf)
Problem(model, obj, x0, tf; [xf, constraints, N, X0, U0, dt, integration])
```
where `Z` is a [`RobotDynamics.SampledTrajectory`].

# Arguments
* `model`: A `DiscreteDynamics` model. If a `ContinuousDynamics` model is provided, it will
           be converted to a `DiscretizedDynamics` model via the integrator specified by the
           `integration` keyword argument.
* `obj`: Objective
* `X0`: Initial state trajectory. If omitted it will be initialized with NaNs, to be later overwritten by the solver.
* `U0`: Initial control trajectory. If omitted it will be initialized with zeros.
* `x0`: Initial state
* `xf`: Final state. Defaults to zeros.
* `dt`: Time step. Can be either a vector of length `N-1` or a positive real number.
* `tf`: Final time. Set to zero.
* `N`: Number of knot points. Uses the length of the objective.
* `integration`: One of the defined integration types to discretize the continuous dynamics model.

Both `X0` and `U0` can be either a `Matrix` or a `Vector{<:AbstractVector}`.
"""
struct Problem{T<:AbstractFloat}
    model::Vector{<:DiscreteDynamics}
    obj::AbstractObjective
    constraints::ConstraintList
    x0::MVector
    xf::MVector
    Z::SampledTrajectory{Nx,Nu,T,KnotPoint{Nx,Nu,Vector{T},T}} where {Nx,Nu}
    N::Int
    function Problem(models::Vector{<:DiscreteDynamics}, obj::AbstractObjective,
            constraints::ConstraintList,
            x0::StaticVector, xf::StaticVector,
            Z::SampledTrajectory{Nx,Nu}, N::Int, t0::T, tf::T) where {Q,T,Nx,Nu} 
        nx,nu = RD.dims(models)
        @assert length(x0) == nx[1]
        @assert length(xf) == nx[end]
        @assert length(Z) == N
        @assert length(models) == N-1 
        @assert tf > t0
        @assert RD.time(Z[1]) ≈ t0
        @assert RD.time(Z[end]) ≈ tf

        # Convert to a trajectory that uses normal Julia vectors
        Z_new = SampledTrajectory(map(1:N) do k
            z = Z[k]
            KnotPoint{Nx,Nu}(nx[k], nu[k], Vector(z.z), RD.time(z), RD.timestep(z))
        end)
        # @assert RobotDynamics.state_dim(obj) == n  "Objective state dimension doesn't match model"
        # @assert RobotDynamics.control_dim(obj) == m "Objective control dimension doesn't match model"
        constraints.nx == nx || throw(DimensionMismatch("Constraint state dimensions don't match model"))
        constraints.nu == nu || throw(DimensionMismatch("Constraint control dimensions don't match model"))
        nx_obj, nu_obj = RD.dims(obj)
        nx_obj == nx || throw(DimensionMismatch("Objective state dimensions don't match model."))
        nu_obj == nu || throw(DimensionMismatch("Objective control dimensions don't match model."))
        # @assert RobotDynamics.dims(Z) == (n,m,N) "Trajectory sizes don't match"
        # TODO: validate trajectory size
        new{T}(models, obj, constraints, x0, xf, Z_new, N)
    end
end

#############################################
# Constructors 
#############################################

function Problem(models::Vector{<:DiscreteDynamics}, obj::O, x0::AbstractVector, tf::Real;
        xf::AbstractVector = fill(NaN, state_dim(models[end])),
        constraints=ConstraintList(models),
        t0=zero(tf),
        X0=[fill(NaN, n) for n in RD.dims(models)[1]],
        U0=[fill(0.0, RD.control_dim(model)) for model in models],
        kwargs...) where {O}

    # Check control dimensions
    nx,nu = RD.dims(models)
    same_state_dimension = all(x->x == nx[1], nx)
    same_control_dimension = all(x->x == nu[1], nu)
    Nx = same_state_dimension ? nx[1] : Any
    Nu = same_control_dimension ? nu[1] : Any
    N = length(obj)
    if X0 isa AbstractMatrix
        X0 = [X0[:,k] for k = 1:size(X0,2)]
    end
    if U0 isa AbstractMatrix
        U0 = [U0[:,k] for k = 1:size(U0,2)]
    end
    Z = SampledTrajectory{Nx,Nu}(X0,U0; tf=tf, kwargs...)
    RD.setinitialtime!(Z, t0)

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

function Problem(p::Problem; model=p.model, obj=copy(p.obj), constraints=copy(p.constraints),
    x0=copy(p.x0), xf=copy(p.xf), t0=p.t0, tf=p.tf)
    Problem(model, obj, constraints, x0, xf, copy(p.Z), p.N, t0, tf)
end

#############################################
# Getters
#############################################

"""
    RobotDynamics.dims(prob)

Return vectors of the state and control dimensions at each time step in the problem.
"""
RD.dims(prob::Problem) = RD.dims(prob.model)

"""
    RobotDynamics.dims(prob, k)

Return `(n,m,N)`, the state and control dimensions at time step `k`, and the length 
of the trajectory, `N`.
"""
RD.dims(prob::Problem, i::Integer) = RD.dims(prob.model[i])..., prob.N 

RD.state_dim(prob::Problem, k::Integer) = state_dim(prob.model[k])
RD.control_dim(prob::Problem, k::Integer) = control_dim(prob.model[k])

"""
    horizonlength(prob::Problem)

Number of knot points in the time horizon, i.e. the length of the sampled 
state and control trajectories.
"""
horizonlength(prob::Problem) = prob.N

import Base.size
@deprecate size(prob::Problem) dims(prob) 

"""
    controls(::Problem, args...)

Get the control trajectory
"""
controls(prob, args...) = controls(get_trajectory(prob), args...)

"""
    states(::Problem, args...)

Get the state trajectory.
"""
states(prob, args...) = states(get_trajectory(prob), args...)

"""
	gettimes(::Problem)

Get the times for all the knot points in the problem.
"""
@inline RobotDynamics.gettimes(prob::Problem) = gettimes(get_trajectory(prob))

"""
    get_initial_time(problem)
    
Get the initial time of the trajectory.
"""
get_initial_time(prob::Problem) = RD.time(get_trajectory(prob)[1])

"""
    get_final_time(problem)

Get the final time of the trajectory.
"""
get_final_time(prob::Problem) = RD.time(get_trajectory(prob)[end])

"Get problem constraints. Returns [`ConstraintList`](@ref)."
@inline get_constraints(prob::Problem) = prob.constraints

num_constraints(prob::Problem) = get_constraints(prob).p

"""
    get_model(prob::Problem)

Get the dynamics models used at each time step. 
Returns Vector{`RobotDynamics.DiscreteDynamics`}.
"""
@inline get_model(prob::Problem) = prob.model

"""
    get_model(prob::Problem, k::Integer)

Get the dynamics model at time step `k`.
"""
@inline get_model(prob::Problem, k) = prob.model[k]

"Get the objective. Returns an `AbstractObjective`."
@inline get_objective(prob::Problem) = prob.obj

"Get the trajectory. Returns an `RobotDynamics.SampledTrajectory`"
@inline get_trajectory(prob::Problem) = prob.Z

"Determines if the problem is constrained."
@inline is_constrained(prob) = isempty(get_constraints(prob))

"Get the in initial state. Returns an `AbstractVector`."
@inline get_initial_state(prob::Problem) = prob.x0

"Get the in initial state. Returns an `AbstractVector`."
@inline get_final_state(prob::Problem) = prob.xf

#############################################
# Setters
#############################################

"""
	initial_trajectory!(prob::Problem, Z)

Copies the trajectory data from `Z` to the problem.
"""
function initial_trajectory!(prob::Problem, Z0::SampledTrajectory)
	Z = get_trajectory(prob)
    copyto!(Z, Z0)
end

"""
	initial_states!(::Problem, X0::Vector{<:AbstractVector})
	initial_states!(::Problem, X0::AbstractMatrix)

Copy the state trajectory
"""
@inline initial_states!(prob, X0) = RobotDynamics.setstates!(get_trajectory(prob), X0)

"""
	initial_controls!(::Problem, U0::Vector{<:AbstractVector})
	initial_controls!(::Problem, U0::AbstractMatrx)

Copy the control trajectory
"""
@inline initial_controls!(prob, U0) = RobotDynamics.setcontrols!(get_trajectory(prob), U0)



"""
	set_initial_state!(prob::Problem, x0::AbstractVector)

Set the initial state in `prob` to `x0`
"""
function set_initial_state!(prob::Problem, x0::AbstractVector)
    prob.x0 .= x0
end

"""
    setinitialtime!(prob, t0)

Set the initial time of the optimization problem, shifting the time of all points in the trajectory.
Returns the updated final time.
"""
function RD.setinitialtime!(prob, t0)
    Z = get_trajectory(prob)
    RD.setinitialtime!(Z, t0)
    Z[end].t
end

@deprecate set_initial_time!(prob, t0::Real) RD.setinitialtime!(prob, t0)

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

#############################################
# Other Methods
#############################################

"""
    cost(::Problem)

Compute the cost for the current trajectory
"""
@inline cost(prob::Problem, Z=prob.Z) = cost(prob.obj, Z)

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

function Base.copy(prob::Problem)
    Problem(prob.model, copy(prob.obj), copy(prob.constraints), copy(prob.x0), copy(prob.xf),
        copy(prob.Z), prob.N, prob.t0, prob.tf)
end
