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
end
"$(TYPEDEF) Create a stage-wise constraint, using ForwardDiff to generate the Jacobian"
function Constraint{S}(c::Function,n::Int,m::Int,p::Int,label::Symbol) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c,∇c,p,label)
end

"$(TYPEDEF) Terminal constraint"
struct TerminalConstraint{S} <: AbstractConstraint{S}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
end
"$(TYPEDEF) Create a terminal constraint, using ForwardDiff to generate the Jacobian"
function TerminalConstraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,p)
    TerminalConstraint{S}(c,∇c,p,label)
end
"$(TYPEDEF) Convenient constructor for terminal constraints"
Constraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintType = TerminalConstraint(c,n,p,label)

"$(SIGNATURES) Return the type of the constraint (Inequality or Equality)"
type(::AbstractConstraint{S}) where S = S
"$(SIGNATURES) Number of element in constraint function (p)"
Base.length(C::AbstractConstraint) = C.p

"""$(SIGNATURES) Create a stage bound constraint
Will default to bounds at infinity. "trim" will remove any bounds at infinity from the constraint function.
"""
function bound_constraint(n::Int,m::Int; x_min=ones(n)*-Inf, x_max=ones(n)*Inf,
                                         u_min=ones(m)*-Inf, u_max=ones(m)*Inf, trim::Bool=false)
     # Validate bounds
     u_max, u_min = _validate_bounds(u_max,u_min,m)
     x_max, x_min = _validate_bounds(x_max,x_min,n)

     # Specify stacking
     inds = create_partition((n,m,n,m),(:x_max,:u_max,:x_min,:u_min))

     # Pre-allocate jacobian
     jac_bnd = [Diagonal(I,n+m); -Diagonal(I,n+m)]
     jac_A = view(jac_bnd,:,inds.x_max)
     jac_B = view(jac_bnd,:,inds.u_max)

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
         Constraint{Inequality}(bound_trim,∇c_trim,count(active_all),:bound)
     else
         # Generate function
         function bound_all(v,x,u)
             v[inds.x_max] = x - x_max
             v[inds.u_max] = u - u_max
             v[inds.x_min] = x_min - x
             v[inds.u_min] = u_min - u
         end
         ∇c(C,x,u) = copyto!(C,jac_bnd)
         Constraint{Inequality}(bound_all,∇c,2(n+m),:bound)
     end
 end

"$(SIGNATURES) Generate a jacobian function for a given in-place function of the form f(v,x)"
function generate_jacobian(f!::Function,n::Int,p::Int=n)
    ∇f!(A,v,x) = ForwardDiff.jacobian!(A,f!,v,x)
    return ∇f!, f!
end

# Constraint Sets
ConstraintSet = Vector{<:AbstractConstraint{S} where S}
StageConstraintSet = Vector{T} where T<:Constraint
TerminalConstraintSet = Vector{T} where T<:TerminalConstraint

"$(SIGNATURES) Count the number of inequality and equality constraints in a constraint set.
Returns the sizes of the constraint vector, not the number of constraint types."
function count_constraints(C::ConstraintSet)
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

"$(SIGNATURES) Evaluate the constraint function for all the stage-wise constraint functions in a set"
function evaluate!(c::BlockVector, C::StageConstraintSet, x, u)
    for con in C
        con.c(c[con.label],x,u)
    end
end
evaluate!(c::BlockVector, C::ConstraintSet, x, u) = evaluate!(c,stage(C),x,u)

"$(SIGNATURES) Evaluate the constraint function for all the terminal constraint functions in a set"
function evaluate!(c::BlockVector, C::TerminalConstraintSet, x)
    for con in C
        con.c(c[con.label],x)
    end
end
evaluate!(c::BlockVector, C::ConstraintSet, x) = evaluate!(c,terminal(C),x)

"$(SIGNATURES) Return number of stage constraints in a set"
num_stage_constraints(C::ConstraintSet) = sum(length.(stage(C)))
"$(SIGNATURES) Return number of stage constraints in a set"
num_terminal_constraints(C::ConstraintSet) = sum(length.(terminal(C)))

RigidBodyDynamics.num_constraints(C::ConstraintSet) = sum(length.(C))
labels(C::ConstraintSet) = [c.label for c in C]
terminal(C::ConstraintSet) = Vector{TerminalConstraint}(filter(x->isa(x,TerminalConstraint),C))
stage(C::ConstraintSet) = Vector{Constraint}(filter(x->isa(x,Constraint),C))
inequalities(C::ConstraintSet) = filter(x->isa(x,AbstractConstraint{Inequality}),C)
equalities(C::ConstraintSet) = filter(x->isa(x,AbstractConstraint{Equality}),C)
bounds(C::ConstraintSet) = filter(x->x.label==:bound,C)
Base.findall(C::ConstraintSet,T::Type) = isa.(C,Constraint{T})
function PartedArrays.create_partition(C::ConstraintSet)
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
end
PartedArrays.BlockArray(C::ConstraintSet) = BlockArray(zeros(sum(length.(C))), create_partition(C))
PartedArrays.BlockArray(T::Type,C::ConstraintSet) = BlockArray(zeros(T,sum(length.(C))), create_partition(C))
