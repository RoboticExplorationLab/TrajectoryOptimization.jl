using PartedArrays, Test, ForwardDiff
using BenchmarkTools
using DocStringExtensions

abstract type ConstraintType end
abstract type Equality <: ConstraintType end
abstract type Inequality <: ConstraintType end
abstract type Null <: ConstraintType end

abstract type GeneralConstraint end
abstract type AbstractConstraint{S<:ConstraintType} <: GeneralConstraint end

import Base.+
function +(C::Vector{<:GeneralConstraint}, con::GeneralConstraint)
    C_ = append!(GeneralConstraint[], C)
    push!(C_, con)
end

function +(C::Vector{<:GeneralConstraint}, C2::Vector{<:GeneralConstraint})
    C_ = append!(GeneralConstraint[], C)
    append!(C_, C2)
end
+(con1::GeneralConstraint, con2::GeneralConstraint) = [con1] + con2

"$(SIGNATURES) Return the type of the constraint (Inequality or Equality)"
type(::AbstractConstraint{S}) where S = S


function con_methods(f::Function)
    term = hasmethod(f, (AbstractVector, AbstractVector))
    stage = hasmethod(f, (Vector, Vector, Vector))
    if term && stage
        :all
    elseif stage
        :stage
    elseif term
        :terminal
    else
        ArgumentError("The function doesn't match the expected signatures")
    end
end

"$(TYPEDEF) General nonlinear constraint"
struct Constraint{S} <: AbstractConstraint{S}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
    inds::Vector{Vector{Int}}
    type::Symbol
    function Constraint{S}(c::Function, ∇c::Function, p::Int, label::Symbol,
            inds::Vector{Vector{Int}}, type::Symbol) where S <: ConstraintType
        if type ∉ [:all,:stage,:terminal]
            ArgumentError(string(type) * " is not a supported constraint type")
        end
        new{S}(c,∇c,p,label,inds,type)
    end
end

"$(TYPEDEF) Create a stage-wise constraint, using ForwardDiff to generate the Jacobian"
function Constraint{S}(c::Function, n::Int, m::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)], term=con_methods(c)) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c, ∇c, p, label, inds, term)
end

"Create a terminal constraint using ForwardDiff"
function Constraint{S}(c::Function, n::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)], term=:terminal) where S<:ConstraintType
    m = 0
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c, ∇c, p, label, inds, term)
end

"$(TYPEDEF) Create a constraint, providing an analytical Jacobian"
function Constraint{S}(c::Function, ∇c::Function, n::Int, m::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)], term=con_methods(c)) where S<:ConstraintType
    Constraint{S}(c, ∇c, p, label, inds, term)
end

"Create a terminal constraint with analytical Jacobian"
function Constraint{S}(c::Function, ∇c::Function, n::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:0)], term=:terminal) where S<:ConstraintType
    Constraint{S}(c, ∇c, p, label, inds, term)
end

function Constraint()
    Constraint{Null}(x->nothing, x->nothing, 0, 0, 0, :null, term)
end

evaluate!(v::AbstractVector, con::Constraint, x::AbstractVector, u::AbstractVector) = is_stage(con) ? con.c(v, x, u) : nothing
evaluate!(v::AbstractVector, con::Constraint, x::AbstractVector) = is_terminal(con) ? con.c(v,x) : nothing
jacobian!(V::AbstractMatrix, con::Constraint, x::AbstractVector, u::AbstractVector) = is_stage(con) ? con.∇c(V, x, u) : nothing
jacobian!(V::AbstractMatrix, con::Constraint, x::AbstractVector) = is_terminal(con) ? con.∇c(V,x) : nothing

is_terminal(con::Constraint) = con.type != :stage
is_stage(con::Constraint) = con.type != :terminal
function Base.length(con::Constraint, type=:stage)
    if type == :stage && is_stage(con)
        return con.p
    elseif type == :terminal && is_terminal(con)
        return con.p
    else
        return 0
    end
end


"$(SIGNATURES) Linear bound constraint on states and controls"
struct BoundConstraint{T} <: AbstractConstraint{Inequality}
    x_max::Vector{T}
    x_min::Vector{T}
    u_max::Vector{T}
    u_min::Vector{T}
    label::Symbol
    active::NamedTuple{(:x_max, :u_max, :x_min, :u_min, :all),NTuple{5,BitArray{1}}}
    inds::Vector{Vector{Int}}
    part::NamedTuple{(:x_max, :u_max, :x_min, :u_min),NTuple{4,UnitRange{Int64}}}
end

"""$(SIGNATURES) Create a stage bound constraint
Will default to bounds at infinity. "trim" will remove any bounds at infinity from the constraint function.
"""
function BoundConstraint(n::Int,m::Int; x_min=ones(n)*-Inf, x_max=ones(n)*Inf,
                                        u_min=ones(m)*-Inf, u_max=ones(m)*Inf, trim::Bool=true)
     # Validate bounds
     u_max, u_min = _validate_bounds(u_max,u_min,m)
     x_max, x_min = _validate_bounds(x_max,x_min,n)

     # Specify stacking
     part = create_partition((n,m,n,m),(:x_max,:u_max,:x_min,:u_min))

     # Pre-allocate jacobian
     jac_bnd = [Diagonal(I,n+m); -Diagonal(I,n+m)]
     jac_A = view(jac_bnd,:,part.x_max)
     jac_B = view(jac_bnd,:,part.u_max)

     inds2 = [collect(1:n), collect(1:m)]

     # Specify which controls and states get used  TODO: Allow this as an input

     if trim
         active = (x_max=isfinite.(x_max),u_max=isfinite.(u_max),
                   x_min=isfinite.(x_min),u_min=isfinite.(u_min))
         lengths = [count(val) for val in values(active)]
         part = create_partition(Tuple(lengths),keys(active))

         active_all = vcat(values(active)...)
         active = merge(active, (all=active_all,))
         ∇c_trim(C,x,u) = copyto!(C,jac_bnd[active_all,:])
     else
         active = (x_max=trues(n),u_max=trues(m),
                   x_min=trues(n),u_min=trues(m),
                   all=trues(2(n+m)))
     end
     return BoundConstraint(x_max, x_min, u_max, u_min, :bound, active, inds2, part)
end

function combine(bnd1::BoundConstraint, bnd2::BoundConstraint)
    n1,m1 = length(bnd1.x_max), length(bnd1.u_max)
    n2,m2 = length(bnd2.x_max), length(bnd2.u_max)
    bnd = BoundConstraint(n1+n2, m1+m2,
                          x_max=[bnd1.x_max; bnd2.x_max],
                          x_min=[bnd1.x_min; bnd2.x_min],
                          u_max=[bnd1.u_max; bnd2.u_max],
                          u_min=[bnd1.u_min; bnd2.u_min])
end

Base.vcat(bnd1::BoundConstraint, bnd2::BoundConstraint) = combine(bnd1, bnd2)

function evaluate!(v::AbstractVector, bnd::BoundConstraint, x::AbstractVector, u::AbstractVector)
    inds = bnd.part
    active = bnd.active
    v[inds.x_max] = (x - bnd.x_max)[active.x_max]
    v[inds.u_max] = (u - bnd.u_max)[active.u_max]
    v[inds.x_min] = (bnd.x_min - x)[active.x_min]
    v[inds.u_min] = (bnd.u_min - u)[active.u_min]
end

function evaluate!(v::AbstractVector, bnd::BoundConstraint, x::AbstractVector)
    inds = bnd.part
    active = bnd.active
    m = count(active.u_max)
    v[inds.x_max     ] = (x - bnd.x_max)[active.x_max]
    v[inds.x_min .- m] = (bnd.x_min - x)[active.x_min]
end

function jacobian!(V::AbstractMatrix, bnd::BoundConstraint, x::AbstractVector, u::AbstractVector)
    n,m = length(bnd.x_max), length(bnd.u_max)
    copyto!(V, [Diagonal(I,n+m); -Diagonal(I,n+m)][bnd.active.all,:])
end

function jacobian!(V::AbstractMatrix, bnd::BoundConstraint, x::AbstractVector)
    n = length(bnd.x_max)
    active = bnd.active
    copyto!(V, [Diagonal(I,n)[active.x_max,:]; -Diagonal(I,n)[active.x_min,:]])
end


function Base.length(bnd::BoundConstraint, type=:stage)
    n,m = length(bnd.x_max), length(bnd.u_max)
    if type == :stage
        count(bnd.active.all)
    elseif type == :terminal
        count(bnd.active.x_max) + count(bnd.active.x_min)
    end
end

is_terminal(::BoundConstraint) = true
is_stage(::BoundConstraint) = true



"""
$(SIGNATURES)
Check max/min bounds for state and control.

Converts scalar bounds to vectors of appropriate size and checks that lengths
are equal and bounds do not result in an empty set (i.e. max > min).

# Arguments
* n: number of elements in the vector (n for states and m for controls)
"""
function _validate_bounds(max,min,n::Int)

    if min isa Real
        min = ones(n)*min
    end
    if max isa Real
        max = ones(n)*max
    end
    if length(max) != length(min)
        throw(DimensionMismatch("u_max and u_min must have equal length"))
    end
    if ~all(max .>= min)
        throw(ArgumentError("u_max must be greater than u_min"))
    end
    if length(max) != n
        throw(DimensionMismatch("limit of length $(length(max)) doesn't match expected length of $n"))
    end
    return max, min
end

function planar_obstacle_constraint(n, m, x_obs, r_obs, label=:obstacle)
    c(v,x,u) = v[1] = circle_constraint(x, x_obs, r_obs)
    # c(v,x) = circle_constraint(x, x_obs, r_obs)
    Constraint{Inequality}(c, n, m, 1, :obstacle)
end


function goal_constraint(xf::Vector{T}) where T
    n = length(xf)
    terminal_constraint(v,xN) = copyto!(v,xN-xf)
    terminal_jacobian(C,xN) = copyto!(C,Diagonal(I,n))
    Constraint{Equality}(terminal_constraint, terminal_jacobian, n, :goal, [collect(1:n),collect(1:0)], :terminal)
end

function infeasible_constraints(n::Int, m::Int)
    idx_inf = (n+m) .+ (1:n)
    u_inf = m .+ (1:n)
    ∇inf = zeros(n,2n+m)
    ∇inf[:,idx_inf] = Diagonal(1.0I,n)
    inf_con(v,x,u) = copyto!(v, u)
    inf_jac(C,x,u) = copyto!(C, ∇inf)
    Constraint{Equality}(inf_con, inf_jac, n, :infeasible, [collect(1:n), collect(u_inf)], :stage)
end



# function generate_jacobian(f!::Function,n::Int,p::Int=n)
#     ∇f!(A,v,x) = ForwardDiff.jacobian!(A,f!,v,x)
#     return ∇f!, f!
# end

########################
#   Constraint Sets    #
########################
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

function RigidBodyDynamics.num_constraints(C::ConstraintSet,type=:stage)
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
function Base.pop!(C::ConstraintSet, label::Symbol)
    con = filter(x->x.label==label,C)
    filter!(x->x.label≠label,C)
    return con[1]
end

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


"$(SIGNATURES) Evaluate the constraint function for all the constraint functions in a set"
function evaluate!(c::PartedVector, C::ConstraintSet, x, u)
    for con in stage(C)
        evaluate!(c[con.label], con, x[con.inds[1]], u[con.inds[2]])
    end
end

"$(SIGNATURES) Evaluate the constraint function for all the terminal constraint functions in a set"
function evaluate!(c::PartedVector, C::ConstraintSet, x)
    for con in terminal(C)
        evaluate!(c[con.label], con, x[con.inds[1]])
    end
end

function jacobian!(c::PartedMatrix, C::ConstraintSet, x, u)
    for con in stage(C)
        jacobian!(c[con.label], con, x, u)
    end
end

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
        _∇c(C,x,u) = con.∇c(view(C,:,idx),x,u)
        _∇c(C,x) = con.∇c(C,x)
        _cs += Constraint{type(con)}(con.c,_∇c,n,m,con.p,con.label,inds=con.inds)
    end

    _cs += bnd

    return _cs
end

# "Type that stores a trajectory of constraint sets"
struct ProblemConstraints
    C::Vector{<:ConstraintSet}
end

function ProblemConstraints(C::ConstraintSet,N::Int)
    C = append!(GeneralConstraint[], C)
    ProblemConstraints([copy(C) for k = 1:N])
end

function ProblemConstraints(C::ConstraintSet,C_term::ConstraintSet,N::Int)
    C = append!(GeneralConstraint[], C)
    ProblemConstraints([k < N ? [C...,TerminalConstraint()] : [Constraint(),C_term...] for k = 1:N])
end

function ProblemConstraints(C::Vector{<:ConstraintSet},C_term::ConstraintSet)
    ProblemConstraints([C...,[Constraint(),C_term...]])
end

function ProblemConstraints(N::Int)
    ProblemConstraints([GeneralConstraint[] for k = 1:N])
end

function ProblemConstraints()
    ProblemConstraints(ConstraintSet[])
end

num_stage_constraints(pcon::ProblemConstraints) = map(num_stage_constraints, pcon.C)
num_terminal_constraints(pcon::ProblemConstraints) = map(num_terminal_constraints, pcon.C)

function TrajectoryOptimization.num_constraints(pcon::ProblemConstraints)
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

Base.setindex!(pcon::ProblemConstraints, C::ConstraintSet, k::Int) = pcon.C[k] = C
Base.getindex(pcon::ProblemConstraints,i::Int) = pcon.C[i]
Base.copy(pcon::ProblemConstraints) = ProblemConstraints(deepcopy(pcon.C))
Base.length(pcon::ProblemConstraints) = length(pcon.C)


"Update constraints trajectories"
function update_constraints!(C::PartedVecTrajectory{T}, constraints::ProblemConstraints,
        X::AbstractVectorTrajectory{T}, U::AbstractVectorTrajectory{T}) where T
    N = length(X)
    for k = 1:N-1
        evaluate!(C[k],constraints[k],X[k],U[k])
    end
    evaluate!(C[N],constraints[N],X[N])
end

function jacobian!(C::PartedMatTrajectory{T}, constraints::ProblemConstraints,
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
