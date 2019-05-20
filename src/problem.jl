"$(TYPEDEF) Trajectory Optimization Problem"
struct Problem{T<:AbstractFloat,D<:DynamicsType}
    model::AbstractModel
    obj::AbstractObjective
    constraints::ProblemConstraints
    x0::Vector{T}
    X::VectorTrajectory{T}
    U::VectorTrajectory{T}
    N::Int
    dt::T
    tf::T

    function Problem(model::Model{M,D}, obj::AbstractObjective, constraints::ProblemConstraints,
        x0::Vector{T}, X::VectorTrajectory, U::VectorTrajectory, N::Int, dt::T, tf::T) where {M,T,D}

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

        new{T,D}(model,obj,constraints,x0,X,U,N,dt,tf)
    end
end

"""$(TYPEDSIGNATURES)
Create a problem from a continuous model, specifying the discretizer as a symbol
"""
function Problem(model::Model{M,Continuous}, obj::AbstractObjective; integration=:rk4, kwargs...) where M
    if integration == :none
        return Problem(model, obj; kwargs...)
    elseif isdefined(TrajectoryOptimization,integration)
        discretizer = eval(integration)
        return Problem(discretizer(model), obj; kwargs...)
    else
        throw(ArgumentError("$integration is not a defined integration scheme"))
    end
end

"""$(TYPEDSIGNATURES)
Create Problem, optionally specifying constraints, initial state, and length.
At least 2 of N, dt, or tf must be specified
"""
function Problem(model::Model, obj::AbstractObjective, X0::VectorTrajectory{T}, U0::VectorTrajectory{T};
        constraints::ProblemConstraints=ProblemConstraints(), x0::Vector{T}=zeros(model.n),
        N::Int=-1, dt=NaN, tf=NaN) where T
    N, tf, dt = _validate_time(N, tf, dt)
    Problem(model, obj, constraints, x0, deepcopy(X0), deepcopy(U0), N, dt, tf)
end
Problem(model::Model, obj::Objective, X0::Matrix{T}, U0::Matrix{T}; kwargs...) where T =
    Problem(model, obj, to_dvecs(X0), to_dvecs(U0); kwargs...)

function Problem(model::Model, obj::AbstractObjective, U0::VectorTrajectory{T};
        constraints::ProblemConstraints=ProblemConstraints(), x0::Vector{T}=zeros(model.n),
        N::Int=-1, dt=NaN, tf=NaN) where T
    N = length(U0) + 1
    N, tf, dt = _validate_time(N, tf, dt)
    X0 = empty_state(model.n, N)
    Problem(model, obj, constraints, x0, deepcopy(X0), deepcopy(U0), N, dt, tf)
end
Problem(model::Model, obj::AbstractObjective, U0::Matrix{T}; kwargs...) where T =
    Problem(model, obj, to_dvecs(U0); kwargs...)

function Problem(model::Model, obj::AbstractObjective;
        constraints::ProblemConstraints=ProblemConstraints(), x0::Vector{T}=zeros(model.n),
        N::Int=-1, dt=NaN, tf=NaN) where T
    N, tf, dt = _validate_time(N, tf, dt)
    X0 = empty_state(model.n, N)
    U0 = [zeros(T,model.m) for k = 1:N-1]
    Problem(model, obj, constraints, x0, X0, U0, N, dt, tf)
end

"$(SIGNATURES) Pass in a cost instead of an objective"
function Problem(model::Model{M,Discrete}, cost::CostFunction, U0::VectorTrajectory{T}; kwargs...) where {M,T}
    N = length(U0) + 1
    obj = Objective(cost, N)
    Problem(model, obj, U0; kwargs...)
end

Problem(model::Model{M,Discrete}, cost::CostFunction, U0::Matrix{T}; kwargs...) where {M,T} =
    Problem(model, cost, to_dvecs(U0); kwargs...)


"$(TYPEDSIGNATURES) Set the initial control trajectory for a problem"
function initial_controls!(prob::Problem{T,D}, U0::AbstractVectorTrajectory{T}) where {T,D}
    m = prob.model.m
    for k = 1:N-1
        prob.U[k][1:m] .= U0[k][1:m]
    end
end

initial_controls!(prob::Problem{T,D}, U0::Matrix{T}) where {T,D} = initial_controls!(prob, to_dvecs(U0))

"$(TYPEDSIGNATURES) Set the initial state trajectory for a problem"
initial_state!(prob::Problem{T,D}, X0::AbstractVectorTrajectory{T}) where {T,D} = copyto!(prob.X, X0)
initial_state!(prob::Problem{T,D}, X0::Matrix{T}) where {T,D} = initial_state!(prob, to_dvecs(X0))

set_x0!(prob::Problem{T,D}, x0::Vector{T}) where {T,D} = copyto!(prob.x0, x0)

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

Base.size(p::Problem)::NTuple{3,Int} = (p.model.n,p.model.m,p.N)

Base.copy(p::Problem) = Problem(p.model, p.obj, copy(p.constraints), copy(p.x0),
    deepcopy(p.X), deepcopy(p.U), p.N, p.dt, p.tf)

empty_state(n::Int,N::Int) = [ones(n)*NaN32 for k = 1:N]

# is_constrained(p::Problem) = !isempty(p.constraints)
is_constrained(prob::Problem{T}) where T = !all(isempty.(prob.constraints.C))

TrajectoryOptimization.num_constraints(prob::Problem) = num_constraints(prob.constraints)

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
function add_constraints!(p::Problem, c::AbstractConstraint)
    push!(p.constraints[1],c)
    return nothing
end

# "$(SIGNATURES) Add a set of constraints to the problem"
# function add_constraints!(p::Problem,C::ConstraintSet)
#     append!(p.constraints,C)
# end

"$(SIGNATURES) Add a set of constraints to the problem"
function add_constraints!(p::Problem,C::ProblemConstraints)
    # append!(p.constraints,C)
    p.constraints .= C
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

num_stage_constraints(p::Problem) = [num_stage_constraints(p.constraints[k]) for k = 1:p.N-1]
num_terminal_constraints(p::Problem) = num_terminal_constraints(p.constraints.C[end])


cost(prob::Problem{T,D}) where {T,D} = cost(prob.obj, prob.X, prob.U)::T

pos(x) = max(0,x)

function max_violation(prob::Problem{T,D}) where {T,D}
    if is_constrained(prob)
        N = prob.N
        c_max = 0.0
        for k = 1:N-1
            if num_stage_constraints(prob.constraints[k]) > 0
                stage_con = stage(prob.constraints[k])
                c = PartedVector(T,stage_con)
                evaluate!(c,stage_con,prob.X[k],prob.U[k])
                max_E = norm(c.equality,Inf)
                max_I = maximum(pos.(c))
                c_max = max(c_max,max(max_E,max_I))
            end
        end
        if num_terminal_constraints(prob.constraints[N]) > 0
            c = PartedVector(T,prob.constraints[N],:terminal)
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

function Expansion(prob::Problem{T,D}) where {T,D}
    n = prob.model.n; m = prob.model.m
    Expansion(zeros(T,n),zeros(T,m),zeros(T,n,n),zeros(T,m,m),zeros(T,m,n))
end

function Expansion(n::Int,m::Int,T::Type)
    Expansion(zeros(T,n),zeros(T,m),zeros(T,n,n),zeros(T,m,m),zeros(T,m,n))
end

function Expansion(prob::Problem{T,D},exp::Symbol) where {T,D}
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
