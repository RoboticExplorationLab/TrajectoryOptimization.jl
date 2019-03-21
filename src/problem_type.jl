"$(TYPEDEF) Trajectory Optimization Problem"
struct Problem{T<:AbstractFloat}
    model::Model
    cost::CostFunction
    constraints::ConstraintSet
    x0::Vector{T}
    X::VectorTrajectory{T}
    U::VectorTrajectory{T}
    N::Int
    dt::T
end

Problem(model::Model,cost::CostFunction) = Problem{Float64}(model,cost,
    AbstractConstraint[],[],Vector[],Vector[],0,0.0)

Problem(T::Type,model::Model,cost::CostFunction) = Problem{T}(model,cost,
    AbstractConstraint[],[],Vector[],Vector[],0,0.0)

Problem(model::Model,cost::CostFunction,x0::Vector{T},U::VectorTrajectory{T}) where T = Problem{T}(model,
    cost,AbstractConstraint[],x0,Vector[],U,length(U)+1,0.0)

Problem(model::Model,cost::CostFunction,x0::Vector{T},U::VectorTrajectory{T},N::Int,dt::T) where T= Problem{T}(model,
    cost,AbstractConstraint[],x0,Vector[],U,N,dt)

Problem(model::Model,cost::CostFunction,x0::Vector{T},X::VectorTrajectory{T},U::VectorTrajectory{T}) where T = Problem{T}(
    model,cost,AbstractConstraint[],x0,X,U,length(X),0.0)

Problem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T}) where T = Problem{T}(model,cost,
    AbstractConstraint[],x0,Vector[],[U[:,k] for k = 1:size(U,2)],size(U,2)+1,0.0)

Problem(model::Model,cost::CostFunction,x0::Vector{T},U::Matrix{T},N::Int,dt::T) where T = Problem{T}(model,cost,
    AbstractConstraint[],x0,Vector[],[U[:,k] for k = 1:size(U,2)],N,dt)

Problem(model::Model,cost::CostFunction,x0::Vector{T},X::Matrix{T},U::Matrix{T}) where T = Problem{T}(
    model,cost,AbstractConstraint[],x0,[X[:,k] for k = 1:size(X,2)],
    [U[:,k] for k = 1:size(U,2)],size(X,2),0.0)

Base.size(p::Problem) = (p.model.n,p.model.m,p.N)

function update_problem(p::Problem;
    model=p.model,cost=p.cost,constraints=p.constraints,x0=p.x0,X=p.X,U=p.U,
    N=p.N,dt=p.dt)

    Problem(model,cost,constraints,x0,X,U,N,dt)
end

function add_constraints!(p::Problem,c::Constraint)
    push!(p.constraints,c)
end

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
