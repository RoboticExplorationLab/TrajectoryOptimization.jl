"$(TYPEDEF) Trajectory Optimization Problem"
struct Problem{T<:AbstractFloat}
    model::Model{Discrete}
    obj::AbstractObjective
    constraints::AbstractConstraintSet
    x0::Vector{T}
    X::VectorTrajectory{T}
    U::VectorTrajectory{T}
    N::Int
    dt::T
    tf::T

    function Problem(model::Model, obj::AbstractObjective, constraints::AbstractConstraintSet,
        x0::Vector{T}, X::VectorTrajectory, U::VectorTrajectory, N::Int, dt::T, tf::T) where T

        n,m = model.n, model.m
        # TODO these checks break for infeasible, minimum time -> do a post check
        # @assert length(X[1]) == n
        # @assert length(U[1]) == m
        # @assert length(x0) == n
        @assert length(X) == N

        if length(U) == N
            @info "Length of U should be N-1, not N. Trimming last entry"
            U = U[1:end-1]
        end
        @assert length(U) == N-1
        if !(dt > 0)
            throw(ArgumentError("dt must be strictly positive"))
        end

        new{T}(model,obj,constraints,x0,X,U,N,dt,tf)
    end
end

"""$(TYPEDSIGNATURES)
Create a problem from a continuous model, specifying the discretizer as a symbol
"""
function Problem(model::Model{Continuous}, obj::AbstractObjective; integration=:rk4, kwargs...)
    if isdefined(TrajectoryOptimization,integration)
        discretizer = eval(integration)
    else
        throw(ArgumentError("$integration is not a defined integration scheme"))
    end
    Problem(discretizer(model), obj; kwargs...)
end

"""$(TYPEDSIGNATURES)
Create Problem, optionally specifying constraints, initial state, and length.
At least 2 of N, dt, or tf must be specified
"""
function Problem(model::Model{Discrete}, obj::AbstractObjective, X0::VectorTrajectory{T}, U0::VectorTrajectory{T};
        constraints::AbstractConstraintSet=AbstractConstraint[], x0::Vector{T}=zeros(model.n),
        N::Int=-1, dt=NaN, tf=NaN) where T
    N, tf, dt = _validate_time(N, tf, dt)
    Problem(model, obj, constraints, x0, X0, U0, N, dt, tf)
end
Problem(model::Model{Discrete}, obj::ObjectiveNew, X0::Matrix{T}, U0::Matrix{T}; kwargs...) where T =
    Problem(model, obj, to_dvecs(X0), to_dvecs(U0); kwargs...)

function Problem(model::Model{Discrete}, obj::AbstractObjective, U0::VectorTrajectory{T};
        constraints::AbstractConstraintSet=AbstractConstraint[], x0::Vector{T}=zeros(model.n),
        N::Int=-1, dt=NaN, tf=NaN) where T
    N = length(U0) + 1
    N, tf, dt = _validate_time(N, tf, dt)
    X0 = empty_state(model.n, N)
    Problem(model, obj, constraints, x0, X0, U0, N, dt, tf)
end
Problem(model::Model{Discrete}, obj::AbstractObjective, U0::Matrix{T}; kwargs...) where T =
    Problem(model, obj, to_dvecs(U0); kwargs...)

function Problem(model::Model{Discrete}, obj::AbstractObjective;
        constraints::AbstractConstraintSet=AbstractConstraint[], x0::Vector{T}=zeros(model.n),
        N::Int=-1, dt=NaN, tf=NaN) where T
    N, tf, dt = _validate_time(N, tf, dt)
    X0 = empty_state(model.n, N)
    U0 = [zeros(T,model.m) for k = 1:N-1]
    Problem(model, obj, constraints, x0, X0, U0, N, dt, tf)
end

"$(TYPEDSIGNATURES) Set the initial control trajectory for a problem"
initial_controls!(prob::Problem{T}, U0::VectorTrajectory{T}) where T = copyto!(prob.U, U0)
initial_controls!(prob::Problem{T}, U0::Matrix{T}) where T = initial_controls!(prob, to_dvecs(U0))

"$(TYPEDSIGNATURES) Set the initial state trajectory for a problem"
initial_state!(prob::Problem{T}, X0::VectorTrajectory{T}) where T = copyto!(prob.X, X0)
initial_state!(prob::Problem{T}, X0::Matrix{T}) where T = initial_state!(prob, to_dvecs(X0))

set_x0!(prob::Problem{T}, x0::Vector{T}) where T = copyto!(prob.x0, x0)

final_time(prob::Problem) = (prob.N-1) * prob.dt

function change_N(prob::Problem, N::Int)
    tf = final_time(prob)
    dt = tf/(N-1)
    X, U = interp_traj(N, tf, prob.X, prob.U)
    @show length(X)
    Problem(prob.model, prob.obj, prob.constraints, prob.x0, X, U, N, dt, tf)
end


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

# Problem(model::Model,cost::CostFunction) = Problem{Float64}(model,cost,
#     AbstractConstraint[],[],Vector[],Vector[],0,0.0)

# Problem(T::Type,model::Model,cost::CostFunction) = Problem{T}(model,cost,
#     AbstractConstraint[],[],Vector[],Vector[],0,0.0)

#
#
# """$(SIGNATURES)
# $(TYPEDSIGNATURES)
# Create a problem, initializing the initial state and control input trajectories to zeros"""
# function Problem(model::Model,cost::CostFunction,N::Int,dt::T) where T
#     X = empty_state(model.n,N)
#     U = [zeros(model.m) for k = 1:N-1]
#     Problem(model,cost,AbstractConstraint[],x0,X,U,N,dt)
# end
#
# """$(SIGNATURES) Create am unconstrained trajectory optimization problem
#
# Creates a problem with discrete dynamics with timestep `dt` from `model`, minimizing the objective given by `cost`, subject
# to an initial state `x0`. `U` is the initial guess for the control trajectory.
# """
# function Problem(model::Model{Discrete},cost::CostFunction,x0::Vector{T},U::VectorTrajectory{T},dt::T) where T
#     N = length(U) + 1
#     X = empty_state(model.n,N)
#     Problem(model,cost,AbstractConstraint[],x0,X,deepcopy(U),N,dt)
# end
# Problem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T},dt::T) where T =
#     Problem(model,cost,x0,to_dvecs(U),dt)

# Problem(model::Model,cost::CostFunction,x0::Vector{T},U::VectorTrajectory{T},N::Int,dt::T) where T= Problem{T}(model,
#     cost,AbstractConstraint[],x0,Vector[],U,N,dt)
#
# Problem(model::Model,cost::CostFunction,x0::Vector{T},X::VectorTrajectory{T},U::VectorTrajectory{T}) where T = Problem{T}(
#     model,cost,AbstractConstraint[],x0,X,U,length(X),0.0)
#
# Problem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T}) where T = Problem{T}(model,cost,
#     AbstractConstraint[],x0,Vector[],[U[:,k] for k = 1:size(U,2)],size(U,2)+1,0.0)
#
# Problem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T},N::Int,dt::T) where T = Problem{T}(model,cost,
#     AbstractConstraint[],x0,Vector[],[U[:,k] for k = 1:size(U,2)],N,dt)
#
# Problem(model::Model,cost::CostFunction,x0::Vector{T},X::Matrix{T},U::Matrix{T}) where T = Problem{T}(
#     model,cost,AbstractConstraint[],x0,[X[:,k] for k = 1:size(X,2)],
#     [U[:,k] for k = 1:size(U,2)],size(X,2),0.0)

Base.size(p::Problem) = (p.model.n,p.model.m,p.N)

Base.copy(p::Problem) = Problem(p.model, p.obj, p.constraints, copy(p.x0),
    deepcopy(p.X), deepcopy(p.U), p.N, p.dt, p.tf)

empty_state(n::Int,N::Int) = [ones(n)*NaN32 for k = 1:N]

is_constrained(p::Problem) = !isempty(p.constraints)

function update_problem(p::Problem;
    model=p.model,obj=p.obj,constraints=p.constraints,x0=p.x0,X=p.X,U=p.U,
    N=p.N,dt=p.dt,tf=p.tf,newProb=true)

    if newProb
        Problem(model,obj,constraints,x0,deepcopy(X),deepcopy(U),N,dt,tf)
    else
        Problem(model,obj,constraints,x0,X,U,N,dt,tf)
    end
end

"$(SIGNATURES) Add a constraint to the problem"
function add_constraints!(p::Problem,c::Constraint)
    push!(p.constraints,c)
end

"$(SIGNATURES) Add a set of constraints to the problem"
function add_constraints!(p::Problem,C::AbstractConstraintSet)
    append!(p.constraints,C)
end

# Problem checks
function check_problem(p::Problem)
    flag = true
    ls = []
    if model.n != length(p.X[1])
        flag = false
        push!(ls,"Model and state trajectory dimension mismatch")
    end
    if model.m != length(p.U[1])
        flag = false
        push!(ls,"Model and control trajectory dimension mismatch")
    end
    if p.N != length(p.X)
        flag = false
        push!(ls,"State trajectory length and knot point mismatch")
    end
    if p.N-1 != length(p.U)
        flag = false
        push!(ls,"Control trajectory length and knot point mismatch")
    end
    if p.N <= 0
        flag = false
        push!(ls,"N <= 0")
    end
    if p.dt <= 0.0
        flag = false
        push!(ls,"dt <= 0")
    end

    if !isempty(ls)
        println("Errors with Problem: ")
        for l in ls
            println(l)
        end
    end

    return flag
end

num_stage_constraints(p::Problem) = num_stage_constraints(p.constraints)
num_terminal_constraints(p::Problem) = num_terminal_constraints(p.constraints)

jacobian!(prob::Problem{T},solver) where T = jacobian!(solver.∇F,prob.model,prob.X,prob.U,prob.dt)

cost(prob::Problem{T}) where T = cost(prob.obj, prob.X, prob.U)::T

pos(x) = max(0,x)

function max_violation(prob::Problem{T}) where T
    if is_constrained(prob)
        N = prob.N
        stage_con = stage(prob.constraints)
        c = BlockVector(T,stage_con)
        c_max = 0.0
        if num_stage_constraints(prob) > 0
            for k = 1:N-1
                evaluate!(c,stage_con,prob.X[k],prob.U[k])
                max_E = norm(c.equality,Inf)
                max_I = maximum(pos.(c))
                c_max = max(c_max,max(max_E,max_I))
            end
        end
        if num_terminal_constraints(prob) > 0
            term_con = terminal(prob.constraints)
            c = BlockVector(T,term_con)
            evaluate!(c,term_con,prob.X[N])
            max_E = norm(c.equality,Inf)
            max_I = maximum(pos.(c))
            c_max = max(c_max,max(max_E,max_I))
        end
        return c_max
    else
        return 0
    end
end

include("infeasible_new.jl")
"Create infeasible state trajectory initialization problem from problem"
function infeasible_problem(prob::Problem{T},R_inf::T=1.0) where T
    # @assert all([prob.obj[k] isa QuadraticCost for k = 1:N])
    # modify problem with slack control
    obj_inf = []
    for k = 1:N-1
        cost_inf = copy(prob.obj[k])
        cost_inf.R = cat(cost_inf.R,R_inf*Diagonal(I,prob.model.n)/prob.dt,dims=(1,2))
        cost_inf.r = [cost_inf.r; zeros(prob.model.n)]
        cost_inf.H = [cost_inf.H; zeros(prob.model.n,prob.model.n)]
        push!(obj_inf,cost_inf)
    end
    push!(obj_inf,copy(prob.obj[N]))

    model_inf = add_slack_controls(prob.model)
    u_slack = slack_controls(prob)
    con_inf = infeasible_constraints(prob.model.n,prob.model.m)

    update_problem(prob,model=model_inf,obj=ObjectiveNew([obj_inf...]),
        constraints=[prob.constraints...,con_inf],U=[[prob.U[k];u_slack[k]] for k = 1:prob.N-1])
end

function minimum_time_problem(prob::Problem{T},R_min_time::T=1.0,dt_max::T=1.0,dt_min::T=1.0e-3) where T
    # modify problem with time step control
    cost_min_time = MinTimeCost(copy(prob.cost),R_min_time)
    model_min_time = add_min_time_controls(prob.model)
    con_min_time_eq, con_min_time_bnd = min_time_constraints(n,m,dt_max,dt_min)
    _con = update_constraint_set_jacobians(prob.constraints,prob.model.n,prob.model.n+1,prob.model.m)

    update_problem(prob,model=model_min_time,obj=cost_min_time,
        constraints=[_con...,con_min_time_eq,con_min_time_bnd],
        U=[[prob.U[k];sqrt(prob.dt)] for k = 1:prob.N-1],
        X=[[prob.X[k];sqrt(prob.dt)] for k = 1:prob.N],
        x0=[x0;0.0])
end
