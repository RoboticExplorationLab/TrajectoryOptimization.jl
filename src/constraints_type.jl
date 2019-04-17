using PartedArrays, Test, ForwardDiff
using BenchmarkTools
using DocStringExtensions

abstract type ConstraintType end
abstract type Equality <: ConstraintType end
abstract type Inequality <: ConstraintType end

abstract type AbstractConstraint{S<:ConstraintType} end

"$(TYPEDEF) Stage constraint"
struct Constraint{S} <: AbstractConstraint{S}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
    inds::Vector{Vector{Int}}
end

"$(TYPEDEF) Create a stage-wise constraint, using ForwardDiff to generate the Jacobian"
function Constraint{S}(c::Function, n::Int, m::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)]) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c, ∇c, p, label, inds)
end

function Constraint{S}(c::Function, ∇c::Function, n::Int, m::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)]) where S<:ConstraintType
    Constraint{S}(c, ∇c, p, label, inds)
end

"$(TYPEDEF) Terminal constraint"
struct TerminalConstraint{S} <: AbstractConstraint{S}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
    inds::Vector{Vector{Int}}
end

"$(TYPEDEF) Create a terminal constraint, using ForwardDiff to generate the Jacobian"
function TerminalConstraint{S}(c::Function, n::Int, p::Int, label::Symbol;
        inds=[collect(1:n)]) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,p)
    TerminalConstraint{S}(c, ∇c, p, label, inds)
end

"$(TYPEDEF) Convenient constructor for terminal constraints"
Constraint{S}(c::Function, n::Int, p::Int, label::Symbol; kwargs...) where S<:ConstraintType =
    TerminalConstraint(c ,n, p, label; kwargs...)

function TerminalConstraint{S}(c::Function, ∇c::Function, n::Int, p::Int, label::Symbol;
        inds=[collect(1:n)]) where S<:ConstraintType
    TerminalConstraint{S}(c, ∇c, p, label, inds)
end

"$(SIGNATURES) Return the type of the constraint (Inequality or Equality)"
type(::AbstractConstraint{S}) where S = S

"$(SIGNATURES) Number of element in constraint function (p)"
Base.length(C::AbstractConstraint) = C.p

"""$(SIGNATURES) Create a stage bound constraint
Will default to bounds at infinity. "trim" will remove any bounds at infinity from the constraint function.
"""
function bound_constraint(n::Int,m::Int; x_min=ones(n)*-Inf, x_max=ones(n)*Inf,
                                         u_min=ones(m)*-Inf, u_max=ones(m)*Inf, trim::Bool=true)
     # Validate bounds
     u_max, u_min = _validate_bounds(u_max,u_min,m)
     x_max, x_min = _validate_bounds(x_max,x_min,n)

     # Specify stacking
     inds = create_partition((n,m,n,m),(:x_max,:u_max,:x_min,:u_min))

     # Pre-allocate jacobian
     jac_bnd = [Diagonal(I,n+m); -Diagonal(I,n+m)]
     jac_A = view(jac_bnd,:,inds.x_max)
     jac_B = view(jac_bnd,:,inds.u_max)

     # Specify which controls and states get used  TODO: Allow this as an input
     inds2 = [collect(1:n), collect(1:m)]

     if trim
         active = (x_max=isfinite.(x_max),u_max=isfinite.(u_max),
                   x_min=isfinite.(x_min),u_min=isfinite.(u_min))
         lengths = [count(val) for val in values(active)]
         inds = create_partition(Tuple(lengths),keys(active))
         function bound_trim(v,x,u)
             v[inds.x_max] = (x - x_max)[active.x_max]
             v[inds.u_max] = (u - u_max)[active.u_max]
             v[inds.x_min] = (x_min - x)[active.x_min]
             v[inds.u_min] = (u_min - u)[active.u_min]
         end
         active_all = vcat(values(active)...)
         ∇c_trim(C,x,u) = copyto!(C,jac_bnd[active_all,:])
         Constraint{Inequality}(bound_trim, ∇c_trim, count(active_all), :bound, inds2)
     else
         # Generate function
         function bound_all(v,x,u)
             v[inds.x_max] = x - x_max
             v[inds.u_max] = u - u_max
             v[inds.x_min] = x_min - x
             v[inds.u_min] = u_min - u
         end
         ∇c(C,x,u) = copyto!(C,jac_bnd)
         Constraint{Inequality}(bound_all, ∇c, 2(n+m), :bound, inds2)
     end
 end

 function bound_constraint(n::Int; x_min=ones(n)*-Inf, x_max=ones(n)*Inf, trim::Bool=false)
    # Validate bounds
    x_max, x_min = _validate_bounds(x_max,x_min,n)

    # Specify stacking
    inds = create_partition((n,n),(:x_max,:x_min))

    # Pre-allocate jacobian
    jac_bnd = [Diagonal(I,n); -Diagonal(I,n)]

    if trim
        active = (x_max=isfinite.(x_max), x_min=isfinite.(x_min))
        lengths = [count(val) for val in values(active)]
        inds = create_partition(Tuple(lengths),keys(active))
        function bound_trim(v,x)
            v[inds.x_max] = (x - x_max)[active.x_max]
            v[inds.x_min] = (x_min - x)[active.x_min]
        end
        active_all = vcat(values(active)...)
        ∇c_trim(C,x,u) = copyto!(C,jac_bnd[active_all,:])
        TerminalConstraint{Inequality}(bound_trim,∇c_trim,count(active_all),:terminal_bound, [collect(1:n)])
    else
        # Generate function
        function bound_all(v,x,u)
            v[inds.x_max] = x - x_max
            v[inds.x_min] = x_min - x
        end
        ∇c(C,x,u) = copyto!(C,jac_bnd)
        TerminalConstraint{Inequality}(bound_all,∇c,2(n+m),:terminal_bound, [collect(1:n)])
    end
end

function goal_constraint(xf::Vector{T}) where T
    n = length(xf)
    terminal_constraint(v,xN) = copyto!(v,xN-xf)
    terminal_jacobian(C,xN) = copyto!(C,Diagonal(I,n))
    TerminalConstraint{Equality}(terminal_constraint, terminal_jacobian, n, :goal, [collect(1:n)])
end

function infeasible_constraints(n::Int, m::Int)
    idx_inf = (n+m) .+ (1:n)
    u_inf = m .+ (1:n)
    ∇inf = zeros(n,2n+m)
    ∇inf[:,idx_inf] = Diagonal(1.0I,n)
    inf_con(v,x,u) = copyto!(v, u)
    inf_jac(C,x,u) = copyto!(C, ∇inf)
    Constraint{Equality}(inf_con, inf_jac, n, :infeasible, [collect(1:n), collect(u_inf)])
end

function min_time_constraints(n::Int,m::Int,dt_max::T=1.0,dt_min::T=1.0e-3) where T
    n̄ = n+1; m̄ = m+1; idx_h = n+m+2
    ∇con_eq = zeros(1,idx_h)
    ∇con_eq[1,idx_h] = 1.0
    ∇con_eq[1,n̄] = -1.0

    function con_eq(v,x,u)
        v[1] = u[end] - x[end]
    end

    jac_eq(C,x,u) = copyto!(C, ∇con_eq)
    con_min_time_eq = Constraint{Equality}(con_eq, jac_eq, 1, :min_time_eq, [collect(1:n̄), collect(1:m̄)])

    ∇con_bnd = zeros(2,idx_h)
    ∇con_bnd[1,idx_h] = 1.0
    ∇con_bnd[2,idx_h] = -1.0

    function con_bnd(v,x,u)
        h = u[end]
        v[1] = h - sqrt(dt_max)
        v[2] = sqrt(dt_min) - h
    end

    jac_bnd(C,x,u) = copyto!(C, ∇con_bnd)

    con_min_time_bnd = Constraint{Inequality}(con_bnd, jac_bnd, 2, :min_time_bnd, [collect(1:n̄), collect(1:m̄)])

    return con_min_time_eq, con_min_time_bnd
end




"$(SIGNATURES) Generate a jacobian function for a given in-place function of the form f(v,x)"
function generate_jacobian(f!::Function,n::Int,p::Int=n)
    ∇f!(A,v,x) = ForwardDiff.jacobian!(A,f!,v,x)
    return ∇f!, f!
end




########################
#   Constraint Sets    #
########################
ConstraintSet = Vector{AbstractConstraint{S} where S}
AbstractConstraintSet = Vector{<:AbstractConstraint{S} where S}
StageConstraintSet = Vector{T} where T<:Constraint
TerminalConstraintSet = Vector{T} where T<:TerminalConstraint

"$(SIGNATURES) Count the number of inequality and equality constraints in a constraint set.
Returns the sizes of the constraint vector, not the number of constraint types."
function count_constraints(C::AbstractConstraintSet)
    pI = 0
    pE = 0
    for c in C
        if c isa AbstractConstraint{Equality}
            pE += c.p
        elseif c isa AbstractConstraint{Inequality}
            pI += c.p
        end
    end
    return pI,pE
end

"$(SIGNATURES) Split a constraint set into sets of inequality and equality constraints"
function Base.split(C::AbstractConstraintSet)
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

"$(SIGNATURES) Evaluate the constraint function for all the stage-wise constraint functions in a set"
function evaluate!(c::BlockVector, C::StageConstraintSet, x, u)
    for con in C
        con.c(c[con.label],x[con.inds[1]],u[con.inds[2]])
    end
end
evaluate!(c::BlockVector, C::AbstractConstraintSet, x, u) = evaluate!(c,stage(C),x,u)

"$(SIGNATURES) Evaluate the constraint function for all the terminal constraint functions in a set"
function evaluate!(c::BlockVector, C::TerminalConstraintSet, x)
    for con in C
        con.c(c[con.label], x[con.inds[1]])
    end
end
evaluate!(c::BlockVector, C::AbstractConstraintSet, x) = evaluate!(c,terminal(C),x)

function jacobian!(Z,C::StageConstraintSet,x::Vector{T},u::Vector{T}) where T
    for con in C
        x_,u_ = x[con.inds[1]], u[con.inds[2]]
        con.∇c(Z[con.label], x_, u_)
    end
end

jacobian!(Z,C::AbstractConstraintSet,x::Vector{T},u::Vector{T}) where T = jacobian!(Z,stage(C),x,u)

function jacobian!(Z,C::TerminalConstraintSet,x::Vector{T}) where T
    for con in C
        con.∇c(Z[con.label], x[con.inds[1]])
    end
end
jacobian!(Z,C::AbstractConstraintSet,x::Vector{T}) where T = jacobian!(Z,terminal(C),x)


function RigidBodyDynamics.num_constraints(C::AbstractConstraintSet)
    if !isempty(C)
        return sum(length.(C))
    else
        return 0
    end
end
labels(C::AbstractConstraintSet) = [c.label for c in C]
terminal(C::AbstractConstraintSet) = Vector{TerminalConstraint}(filter(x->isa(x,TerminalConstraint),C))
stage(C::AbstractConstraintSet) = Vector{Constraint}(filter(x->isa(x,Constraint),C))
inequalities(C::AbstractConstraintSet) = filter(x->isa(x,AbstractConstraint{Inequality}),C)
equalities(C::AbstractConstraintSet) = filter(x->isa(x,AbstractConstraint{Equality}),C)
bounds(C::AbstractConstraintSet) = filter(x->x.label ∈ [:terminal_bound,:bound],C)
Base.findall(C::AbstractConstraintSet,T::Type) = isa.(C,Constraint{T})
function split_bounds(C::AbstractConstraintSet)
    bnd = bounds(C)
    notbnd = setdiff(C, bnd)
    return bnd, notbnd
end
function PartedArrays.create_partition(C::AbstractConstraintSet)
    if !isempty(C)
        lens = length.(C)
        part = create_partition(Tuple(lens),Tuple(labels(C)))
        ineq = BlockArray(trues(sum(lens)),part)
        for c in C
            if type(c) == Equality
                copyto!(ineq[c.label], falses(length(c)))
            end
        end
        part_IE = (inequality=LinearIndices(ineq)[ineq],equality=LinearIndices(ineq)[.!ineq])
        return merge(part,part_IE)
    else
        return NamedTuple{(:equality,:inequality)}((1:0,1:0))
    end
end
function PartedArrays.create_partition2(C::AbstractConstraintSet,n::Int,m::Int)
    if !isempty(C)
        lens = Tuple(length.(C))
        names = Tuple(labels(C))
        p = num_constraints(C)
        part1 = create_partition(lens,names)
        part2 = NamedTuple{names}([(rng,1:n+m) for rng in part1])
        part_xu = (x=(1:p,1:n),u=(1:p,n+1:n+m))
        return merge(part2,part_xu)
    else
        return NamedTuple{(:x,:u)}(((1:0,1:n),(1:0,n+1:n+m)))
    end
end

PartedArrays.BlockVector(C::AbstractConstraintSet) = BlockArray(zeros(num_constraints(C)), create_partition(C))
PartedArrays.BlockVector(T::Type,C::AbstractConstraintSet) = BlockArray(zeros(T,num_constraints(C)), create_partition(C))
PartedArrays.BlockMatrix(C::AbstractConstraintSet,n::Int,m::Int) = BlockArray(zeros(num_constraints(C),n+m), create_partition2(C,n,m))
PartedArrays.BlockMatrix(T::Type,C::AbstractConstraintSet,n::Int,m::Int) = BlockArray(zeros(T,num_constraints(C),n+m), create_partition2(C,n,m))

num_stage_constraints(C::AbstractConstraintSet) = num_constraints(stage(C))
num_terminal_constraints(C::AbstractConstraintSet) = num_constraints(terminal(C))
count_stage_constraints(C::AbstractConstraintSet) = count_constraints(stage(C))
count_terminal_constraints(C::AbstractConstraintSet) = count_constraints(terminal(C))

"Return a new constraint set with modified jacobians--useful for state augmented problems"
function update_constraint_set_jacobians(cs::AbstractConstraintSet,n::Int,n̄::Int,m::Int)
    idx = [(1:n)...,((1:m) .+ n̄)...]
    _cs = []

    for con in stage(cs)
        _∇c(C,x,u) = con.∇c(view(C,:,idx),x,u)
        push!(_cs,Constraint{type(con)}(con.c,_∇c,n,m,con.p,con.label,inds=con.inds))
    end

    for con in terminal(cs)
        push!(_cs,con)
    end

    return [_cs...]
end
