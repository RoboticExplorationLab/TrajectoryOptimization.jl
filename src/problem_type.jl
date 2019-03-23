"$(TYPEDEF) Trajectory Optimization Problem"
struct Problem{T<:AbstractFloat}
    model::Model{Discrete}
    cost::CostFunction
    constraints::ConstraintSet
    x0::Vector{T}
    X::VectorTrajectory{T}
    U::VectorTrajectory{T}
    N::Int
    dt::T

    function Problem(model::Model, cost::CostFunction, constraints::ConstraintSet,
        x0::Vector{T}, X::VectorTrajectory, U::VectorTrajectory, N::Int, dt::T) where T

        n,m = model.n, model.m
        @assert length(X[1]) == n
        @assert length(U[1]) == m
        @assert length(x0) == n
        @assert length(X) == N

        if length(U) == N
            @info "Length of U should be N-1, not N. Trimming last entry"
            U = U[1:end-1]
        end
        @assert length(U) == N-1
        if !(dt > 0)
            throw(ArgumentError("dt must be strictly positive"))
        end

        new{T}(model,cost,constraints,x0,X,U,N,dt)
    end
end



# Problem(model::Model,cost::CostFunction) = Problem{Float64}(model,cost,
#     AbstractConstraint[],[],Vector[],Vector[],0,0.0)

# Problem(T::Type,model::Model,cost::CostFunction) = Problem{T}(model,cost,
#     AbstractConstraint[],[],Vector[],Vector[],0,0.0)

"$(SIGNATURES)
$(TYPEDSIGNATURES)
Create a problem, initializing the initial state and control input trajectories to zeros"
function Problem(model::Model,cost::CostFunction,N::Int,dt::T) where T
    X = empty_state(model.n,N)
    U = [zeros(model.m) for k = 1:N-1]
    Problem(model,cost,AbstractConstraint[],x0,X,U,N,dt)
end

"$(SIGNATURES) Create am unconstrained trajectory optimization problem

Creates a problem with discrete dynamics with timestep `dt` from `model`, minimizing the objective given by `cost`, subject
to an initial state `x0`. `U` is the initial guess for the control trajectory.
"""
function Problem(model::Model{Discrete},cost::CostFunction,x0::Vector{T},U::VectorTrajectory{T},dt::T) where T
    N = length(U) + 1
    X = empty_state(model.n,N)
    Problem(model,cost,AbstractConstraint[],x0,X,deepcopy(U),N,dt)
end
Problem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T},dt::T) =
    Problem(model,cost,x0,to_dvecs(U),dt)

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

Base.copy(p::Problem) = Problem(p.model, p.cost, p.constraints, copy(p.x0),
    deepcopy(p.X), deepcopy(p.U), p.N, p.dt)

empty_state(n::Int,N::Int) = [ones(n)*NaN32 for k = 1:N]

function update_problem(p::Problem;
    model=p.model,cost=p.cost,constraints=p.constraints,x0=p.x0,X=p.X,U=p.U,
    N=p.N,dt=p.dt,newProb=true)

    if newProb
        Problem(model,cost,constraints,x0,deepcopy(X),deepcopy(U),N,dt)
    else
        Problem(model,cost,constraints,x0,X,U,N,dt)
    end
end

"$(SIGNATURES) Add a constraint to the problem"
function add_constraints!(p::Problem,c::Constraint)
    push!(p.constraints,c)
end

"$(SIGNATURES) Add a set of constraints to the problem"
function add_constraints!(p::Problem,C::ConstraintSet)
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


jacobian!(prob::Problem{T},solver) where T = jacobian!(solver.∇F,prob.model,prob.X,prob.U,prob.dt)

cost(prob::Problem,X=prob.X,U=prob.U) = cost(prob.cost,X,U,prob.dt)
