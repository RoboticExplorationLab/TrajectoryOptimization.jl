export
    set_x0!,
    is_constrained,
    update_problem


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
struct Problem{T<:AbstractFloat,D<:DynamicsType}
    model::AbstractModel
    obj::AbstractObjective
    constraints::Constraints
    x0::Vector{T}
    xf::Vector{T}
    X::VectorTrajectory{T}
    U::VectorTrajectory{T}
    N::Int
    dt::T
    tf::T

    function Problem(model::Model{M,D}, obj::AbstractObjective, constraints::Constraints,
        x0::Vector{T},xf::Vector{T}, X::VectorTrajectory, U::VectorTrajectory, N::Int, dt::Real, tf::Real) where {M,T,D}

        n,m = model.n, model.m
        # TODO these checks break for infeasible, minimum time -> do a post check
        # @assert length(X[1]) == n
        # @assert length(U[1]) == m
        # @assert length(x0) == n
        @assert length(obj) == N
        @assert length(constraints) == N
        @assert length(X) == N

        if length(U) == N
            @info "Length of U should be N-1, not N. Trimming last entry"
            U = U[1:end-1]
        end
        @assert length(U) == N-1
        if !(dt > 0)
            throw(ArgumentError("dt must be strictly positive"))
        end

        new{T,D}(model,obj,constraints,x0,xf,X,U,N,dt,tf)
    end
end

function Problem(model::Model{M,Continuous}, obj::AbstractObjective, X0::VectorTrajectory{T}, U0::VectorTrajectory{T};
        N::Int=length(obj), constraints::Constraints=Constraints(N),
        x0::Vector{T}=zeros(model.n), xf::Vector{T}=zeros(model.n),
        dt=NaN, tf=NaN, integration=:none) where {M,T}
    N, tf, dt = _validate_time(N, tf, dt)
    if integration == :none
        model = model
    elseif isdefined(TrajectoryOptimization,integration)
        discretizer = eval(integration)
        model = discretizer(model)
    else
        throw(ArgumentError("$integration is not a defined integration scheme"))
    end
    Problem(model, obj, constraints, x0, xf, deepcopy(X0), deepcopy(U0), N, dt, tf)
end

function Problem(model::Model{M,Discrete}, obj::AbstractObjective, X0::VectorTrajectory{T}, U0::VectorTrajectory{T};
        N::Int=length(obj), constraints::Constraints=Constraints(N), x0::Vector{T}=zeros(model.n),
        xf::Vector{T}=zeros(model.n), dt=NaN, tf=NaN) where {M,T}
    N, tf, dt = _validate_time(N, tf, dt)
    Problem(model, obj, constraints, x0, xf, deepcopy(X0), deepcopy(U0), N, dt, tf)
end

Problem(model::Model, obj::Objective, X0::Matrix{T}, U0::Matrix{T}; kwargs...) where T =
    Problem(model, obj, to_dvecs(X0), to_dvecs(U0); kwargs...)

function Problem(model::Model, obj::AbstractObjective, U0::VectorTrajectory{T}; kwargs...) where T
    N = length(obj)
    X0 = empty_state(model.n, N)
    Problem(model, obj, X0, U0; kwargs...)
end
Problem(model::Model, obj::AbstractObjective, U0::Matrix{T}; kwargs...) where T =
    Problem(model, obj, to_dvecs(U0); kwargs...)

function Problem(model::Model, obj::AbstractObjective; kwargs...)
    N = length(obj)
    U0 = [zeros(model.m) for k = 1:N-1]
    X0 = empty_state(model.n, N)
    Problem(model, obj, X0, U0; kwargs...)
end

# QUESTION: Remove these constructors?
"$(SIGNATURES) Pass in a cost instead of an objective"
function Problem(model::Model{Nominal,Discrete}, cost::CostFunction, U0::VectorTrajectory{T}; kwargs...) where T
    N = length(U0) + 1
    obj = Objective(cost, N)
    Problem(model, obj, U0; kwargs...)
end

Problem(model::Model{Nominal,Discrete}, cost::CostFunction, U0::Matrix{T}; kwargs...) where T =
    Problem(model, cost, to_dvecs(U0); kwargs...)


"""```julia
update_problem(prob; kwargs...)
```
Create a new problem from another, specifing all fields as keyword arguments
The `newProb` argument can be set to true if a the primal variables are to be copied, otherwise they will be passed to the modified problem.
"""
function update_problem(p::Problem;
    model=p.model, obj=p.obj, constraints=p.constraints, x0=p.x0, xf=p.xf, X=p.X, U=p.U,
    N=p.N, dt=p.dt, tf=p.tf, newProb=true)

    if newProb
        Problem(model,obj,constraints,x0,xf,deepcopy(X),deepcopy(U),N,dt,tf)
    else
        Problem(model,obj,constraints,x0,xf,X,U,N,dt,tf)
    end
end

"$(SIGNATURES) Set the initial control trajectory for a problem. U0 can be either a `Matrix` or a `Vector{Vector}`"
initial_controls!(prob::Problem{T}, U0::AbstractVectorTrajectory{T}) where T = copyto!(prob.U, U0[1:prob.N-1])
initial_controls!(prob::Problem{T}, U0::Matrix{T}) where T = initial_controls!(prob, to_dvecs(U0))

"$(SIGNATURES) Set the initial state trajectory for a problem. X0 can be either a `Matrix` or a `Vector{Vector}`"
initial_states!(prob::Problem{T}, X0::AbstractVectorTrajectory{T}) where T = copyto!(prob.X, X0)
initial_states!(prob::Problem{T}, X0::Matrix{T}) where T = initial_state!(prob, to_dvecs(X0))

"$(SIGNATURES) Set the initial state"
set_x0!(prob::Problem{T}, x0::Vector{T}) where T = copyto!(prob.x0, x0)

# TODO: think about how to do this properly now that objective and constraints depend on N
# function change_N(prob::Problem, N::Int)
#     tf = final_time(prob)
#     dt = tf/(N-1)
#     X, U = interp_traj(N, tf, prob.X, prob.U)
#     @show length(X)
#     Problem(prob.model, prob.obj, prob.constraints, prob.x0, X, U, N, dt, tf)
# end


function _validate_time(N,tf,dt)
    if N == -1 && isnan(dt) && isnan(tf)
        throw(ArgumentError("Must specify at least 2: N, dt, or tf"))
    end
    # Check for minimum time
    if (tf == 0) || (tf == :min)
        tf = 0.0
        if N==-1
            throw(ArgumentError("N must be specified for a minimum-time problem"))
        end
    elseif tf > 0
        # Handle combination of N and dt
        if isnan(dt) && N > 0
            dt = tf / (N-1)
        elseif ~isnan(dt) && N == -1
            N, dt = calc_N(tf, dt)
        elseif isnan(dt) && N==-1
            @warn "Neither dt or N were specified. Setting N = 51"
            N = 51
            dt = tf / (N-1)
        elseif ~isnan(dt) && N > 0
            if dt !== tf/(N-1)
                throw(ArgumentError("Specified time step, number of knot points, and final time do not agree ($dt ≢ $(obj.tf)/$(N-1))"))
            end
        end
        if dt == 0
            throw(ArgumentError("dt must be non-zero for non-minimum time problems"))
        end
    elseif isnan(tf)
        if dt > 0
            if N == -1
                @warn "Neither tf or N were specified. Setting N = 51"
                N = 51
            end
            tf = dt*(N-1)
        else
            throw(ArgumentError("dt must be positive for a non-minimum-time problem"))
        end
    else
        throw(ArgumentError("Invalid input for tf"))
    end

    # Check N, dt for valid entries
    if N < 0
        err = ArgumentError("$N is not a valid entry for N. Number of knot points must be a positive integer.")
        throw(err)
    elseif dt < 0
        err = ArgumentError("$dt is not a valid entry for dt. Time step must be positive.")
        throw(err)
    end
    return N,tf,dt
end

"""```
n,m,N = size(p::Problem)
```
Return the number of states (n), number of controls (m), and the number of knot points (N)"""
Base.size(p::Problem)::NTuple{3,Int} = (p.model.n,p.model.m,p.N)

"$(TYPEDSIGNATURES) Copy a problem"
Base.copy(p::Problem) = Problem(p.model, p.obj, copy(p.constraints), copy(p.x0), copy(p.xf),
    deepcopy(p.X), deepcopy(p.U), p.N, p.dt, p.tf)

empty_state(n::Int,N::Int) = [ones(n)*NaN32 for k = 1:N]

"$(SIGNATURES) Checks if a problem has any constraints"
is_constrained(prob::Problem{T}) where T = !all(isempty.(prob.constraints.C))

TrajectoryOptimization.num_constraints(prob::Problem) = num_constraints(prob.constraints)

"$(TYPEDSIGNATURES) Evaluate the current cost for the problem"
cost(prob::Problem{T}) where T = cost(prob.obj, prob.X, prob.U, get_dt_traj(prob))::T

"$(TYPEDSIGNATURES) Evaluate the maximum contraint violation"
function max_violation(prob::Problem{T})::T where T
    if is_constrained(prob)
        N = prob.N
        c_max = 0.0
        for k = 1:N-1
            if num_stage_constraints(prob.constraints[k]) > 0
                stage_con = stage(prob.constraints[k])
                c = PartedVector(stage_con)
                evaluate!(c,stage_con,prob.X[k],prob.U[k])
                max_E = norm(c.equality,Inf)
                max_I = maximum(pos.(c))
                c_max = max(c_max,max(max_E,max_I))
            end
        end
        if num_terminal_constraints(prob.constraints[N]) > 0
            c = PartedVector(prob.constraints[N],:terminal)
            evaluate!(c,prob.constraints[N],prob.X[N])
            max_E = norm(c.equality,Inf)
            max_I = maximum(pos.(c))
            c_max = max(c_max,max(max_E,max_I))
        end
        return c_max
    else
        return 0.
    end
end

function Expansion(prob::Problem{T}) where T
    n = prob.model.n; m = prob.model.m
    Expansion(zeros(T,n),zeros(T,m),zeros(T,n,n),zeros(T,m,m),zeros(T,m,n))
end

function Expansion(prob::Problem{T},exp::Symbol) where T
    n = prob.model.n; m = prob.model.m
    if exp == :x
        return Expansion(zeros(T,n),zeros(T,0),zeros(T,n,n),zeros(T,0,0),zeros(T,0,0))
    elseif exp == :u
        return Expansion(zeros(T,0),zeros(T,m),zeros(T,0,0),zeros(T,m,m),zeros(T,0,0))
    else
        error("Invalid expansion components requested")
    end
end

midpoint(prob::Problem{T,Continuous}) where T = update_problem(prob, model=midpoint(prob.model))
rk3(prob::Problem{T,Continuous}) where T = update_problem(prob, model=rk3(prob.model))
rk4(prob::Problem{T,Continuous}) where T = update_problem(prob, model=rk4(prob.model))

"Return timestep"
function get_dt(prob::Problem,k::Int)
    if prob.tf == 0.
        return prob.U[k][end]^2
    else
        return prob.dt
    end
end

function get_dt(prob::Problem,U::AbstractArray,k::Int)
    if prob.tf == 0.
        return U[end]^2
    else
        return prob.dt
    end
end

function get_dt_traj(prob::Problem)
    [get_dt(prob,k) for k = 1:prob.N-1]
end

function get_dt_traj(prob::Problem,U::Trajectory)
    [get_dt(prob,U[k],k) for k = 1:prob.N-1]
end

"$(SIGNATURES) Get the total time for tje trajectory (applicable for time-penalized problems)"
final_time(prob::Problem) = sum(get_dt_traj(prob))
