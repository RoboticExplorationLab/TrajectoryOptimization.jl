

ConstraintSet = Vector{<:GeneralConstraint}

"$(SIGNATURES) Count the number of inequality and equality constraints in a constraint set.
Returns the sizes of the constraint vector, not the number of constraint types."
function count_constraints(C::ConstraintSet)
    pI = 0
    pE = 0
    for c in C
        if c isa AbstractConstraint{Equality}
            pE += length(c)
        elseif c isa AbstractConstraint{Inequality}
            pI += length(c)
        end
    end
    return pI,pE
end

"$(SIGNATURES) Split a constraint set into sets of inequality and equality constraints"
function Base.split(C::ConstraintSet)
    E = AbstractConstraint{Equality}[]
    I = AbstractConstraint{Inequality}[]
    for c in C
        if c isa AbstractConstraint{Equality}
            push!(E,c)
        elseif c isa AbstractConstraint{Inequality}
            push!(I,c)
        end
    end
    return I,E
end

"$(SIGNATURES) Count the total number of constraints in constraint set `C`. Stage or terminal is specified by `type`"
function TrajectoryOptimization.num_constraints(C::ConstraintSet,type=:stage)
    if !isempty(C)
        return sum(length.(C,type))
    else
        return 0
    end
end

labels(C::ConstraintSet) = [c.label for c in C]
inequalities(C::ConstraintSet) = filter(x->isa(x,AbstractConstraint{Inequality}),C)
equalities(C::ConstraintSet) = filter(x->isa(x,AbstractConstraint{Equality}),C)
bounds(C::ConstraintSet) = filter(x->isa(x,BoundConstraint),C)
Base.findall(C::ConstraintSet,T::Type) = isa.(C,AbstractConstraint{T})
terminal(C::ConstraintSet) = filter(is_terminal,C)
stage(C::ConstraintSet) = filter(is_stage,C)

function remove_bounds!(C::ConstraintSet)
    bnds = bounds(C)
    filter!(x->!isa(x,BoundConstraint),C)
    return bnds
end
"$(SIGNATURES) Remove a bound from a ConstraintSet given its label"
function Base.pop!(C::ConstraintSet, label::Symbol)
    con = filter(x->x.label==label,C)
    filter!(x->x.label≠label,C)
    return con[1]
end

"$(SIGNATURES) Generate the partition for splitting up the combined constraint vector into individual constraints."
function PartedArrays.create_partition(C::ConstraintSet,contype=:stage)
    if !isempty(C)
        lens = length.(C,contype)
        part = create_partition(Tuple(lens),Tuple(labels(C)))
        ineq = PartedArray(trues(sum(lens)),part)
        for (i,c) in enumerate(C)
            if type(c) == Equality
                copyto!(ineq[c.label], falses(lens[i]))
            end
        end
        part_IE = (inequality=LinearIndices(ineq)[ineq],equality=LinearIndices(ineq)[.!ineq])
        return merge(part,part_IE)
    else
        return NamedTuple{(:equality,:inequality)}((1:0,1:0))
    end
end

function PartedArrays.create_partition2(C::ConstraintSet,n::Int,m::Int,contype=:stage)
    if !isempty(C)
        lens = Tuple(length.(C,contype))
        names = Tuple(labels(C))
        p = num_constraints(C,contype)
        part1 = create_partition(lens,names)
        contype == :stage ? m = m : m = 0
        part2 = NamedTuple{names}([(rng,1:n+m) for rng in part1])
        part_xu = (x=(1:p,1:n),u=(1:p,n+1:n+m))
        return merge(part2,part_xu)
    else
        return NamedTuple{(:x,:u)}(((1:0,1:n),(1:0,n+1:n+m)))
    end
end

PartedArrays.PartedVector(C::ConstraintSet, type=:stage) = PartedVector(Float64,C,type)
PartedArrays.PartedVector(T::Type,C::ConstraintSet, type=:stage) = PartedArray(zeros(T,num_constraints(C,type)), create_partition(C,type))
PartedArrays.PartedMatrix(C::ConstraintSet,n::Int,m::Int, type=:stage) = PartedMatrix(Float64, C, n, m, type)
PartedArrays.PartedMatrix(T::Type,C::ConstraintSet,n::Int,m::Int, type=:stage) = PartedArray(zeros(T,num_constraints(C,type),n+m*(type==:stage)), create_partition2(C,n,m,type))

num_stage_constraints(C::ConstraintSet) = num_constraints(stage(C),:stage)
num_terminal_constraints(C::ConstraintSet) = num_constraints(terminal(C),:terminal)


"$(SIGNATURES) Evaluate the constraint function for all the constraint functions in set `C`"
function evaluate!(c::PartedVector, C::ConstraintSet, x, u)
    for con in stage(C)
        evaluate!(c[con.label], con, x[con.inds[1]], u[con.inds[2]])
    end
end

"$(SIGNATURES) Evaluate the constraint function for all the terminal constraint functions in set `C`"
function evaluate!(c::PartedVector, C::ConstraintSet, x)
    for con in terminal(C)
        evaluate!(c[con.label], con, x[con.inds[1]])
    end
end

"$(SIGNATURES) Compute the constraint Jacobian of ConstraintSet `C`"
function jacobian!(c::PartedMatrix, C::ConstraintSet, x, u)
    for con in stage(C)
        jacobian!(c[con.label], con, x, u)
    end
end

"$(SIGNATURES) Compute the constraint Jacobian of ConstraintSet `C` at the terminal time step"
function jacobian!(c::PartedMatrix, C::ConstraintSet, x)
    for con in terminal(C)
        jacobian!(c[con.label], con, x)
    end
end


"Return a new constraint set with modified jacobians--useful for state augmented problems"
function update_constraint_set_jacobians(cs::ConstraintSet,n::Int,n̄::Int,m::Int)
    idx = [collect(1:n); collect(1:m) .+ n̄]
    _cs = GeneralConstraint[]

    cs_ = copy(cs)
    bnd = remove_bounds!(cs_)
    for con in cs_
        _∇c(C,x,u) = con.∇c(view(C,:,idx),x[con.inds[1]],u[con.inds[2]])
        _∇c(C,x) = con.∇c(C,x[con.inds[1]])
        _cs += Constraint{type(con)}(con.c,_∇c,n,m,con.p,con.label,inds=con.inds,inputs=con.inputs)
    end

    _cs += bnd

    return _cs
end

# "Type that stores a trajectory of constraint sets"
"""$(TYPEDEF)
Collection of constraints for a trajectory optimization problem.
    Essentially a list of `ConstraintSets` for each time step
"""
struct Constraints
    C::Vector{<:ConstraintSet}
end

"""Copy a ConstraintSet over all time steps"""
function Constraints(C::ConstraintSet,N::Int)
    C = append!(GeneralConstraint[], C)
    Constraints([copy(C) for k = 1:N])
end

"""Copy a ConstraintSet over all stage time steps, with a unique terminal constraint set"""
function Constraints(C::ConstraintSet,C_term::ConstraintSet,N::Int)
    C = append!(GeneralConstraint[], C)
    C_term = append!(GeneralConstraint[], C_term)
    Constraints([k < N ? C : C_term for k = 1:N])
end

"""Create an empty set of Constraints for a problem with size N"""
function Constraints(N::Int)
    Constraints([GeneralConstraint[] for k = 1:N])
end

function Constraints()
    Constraints(ConstraintSet[])
end

num_stage_constraints(pcon::Constraints) = map(num_stage_constraints, pcon.C)
num_terminal_constraints(pcon::Constraints) = map(num_terminal_constraints, pcon.C)

"""$(SIGNATURES)
Count the number of constraints at each time step. Returns a `Vector{Int}`
"""
function TrajectoryOptimization.num_constraints(pcon::Constraints)::Vector{Int}
    N = length(pcon.C)
    p = zeros(Int,N)
    for k = 1:N-1
        for con in pcon.C[k]
            p[k] += length(con)
        end
    end
    for con in pcon.C[N]
        p[N] += length(con,:terminal)
    end
    return p
end

Base.setindex!(pcon::Constraints, C::ConstraintSet, k::Int) = pcon.C[k] = C
Base.getindex(pcon::Constraints,i::Int) = pcon.C[i]
Base.copy(pcon::Constraints) = Constraints(deepcopy(pcon.C))
Base.length(pcon::Constraints) = length(pcon.C)

function has_bounds(C::Constraints)
    for k = 1:length(C)
        for con in C[k]
            if con isa BoundConstraint
                return true
            end
        end
    end
    return false
end


"Update constraints trajectories from Constraints"
function update_constraints!(C::PartedVecTrajectory{T}, constraints::Constraints,
        X::AbstractVectorTrajectory{T}, U::AbstractVectorTrajectory{T}) where T
    N = length(X)
    for k = 1:N-1
        evaluate!(C[k],constraints[k],X[k],U[k])
    end
    evaluate!(C[N],constraints[N],X[N])
end

"Compute constraint Jacobians from Constraints"
function jacobian!(C::PartedMatTrajectory{T}, constraints::Constraints,
        X::AbstractVectorTrajectory{T}, U::AbstractVectorTrajectory{T}) where T
    N = length(X)
    for k = 1:N-1
        jacobian!(C[k],constraints[k],X[k],U[k])
    end
    jacobian!(C[N],constraints[N],X[N])
end



###################################
##         ACTIVE SET            ##
###################################

"Evaluate active set constraints for entire trajectory"
function update_active_set!(a::PartedVecTrajectory{Bool},c::PartedVecTrajectory{T},λ::PartedVecTrajectory{T},tol::T=0.0) where T
    N = length(c)
    for k = 1:N
        active_set!(a[k], c[k], λ[k], tol)
    end
end

"Evaluate active set constraints for a single time step"
function active_set!(a::AbstractVector{Bool}, c::AbstractVector{T}, λ::AbstractVector{T}, tol::T=0.0) where T
    # inequality_active!(a,c,λ,tol)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return nothing
end

function active_set(c::AbstractVector{T}, λ::AbstractVector{T}, tol::T=0.0) where T
    a = PartedArray(trues(length(c)),c.parts)
    a.equality .= true
    a.inequality .=  @. (c.inequality >= tol) | (λ.inequality > 0)
    return a
end
