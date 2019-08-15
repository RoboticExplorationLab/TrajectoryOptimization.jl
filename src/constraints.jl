using PartedArrays, Test, ForwardDiff
using BenchmarkTools
using DocStringExtensions

export
    violation!,
    label

"Sense of a constraint (inequality / equality / null)"
abstract type ConstraintType end
"Inequality constraints"
abstract type Equality <: ConstraintType end
"Equality constraints"
abstract type Inequality <: ConstraintType end
"An empty constraint"
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

label(con::AbstractConstraint) = con.label

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

"""$(TYPEDEF) General nonlinear vector-valued constraint with `p` entries at a single timestep for `n` states and `m` controls.
# Constructors
```julia
Constraint{S}(confun, ∇confun, n, m, p, label; inds, term, inputs)
Constraint{S}(confun, ∇confun, n,    p, label; inds, term, inputs)  # Terminal constraint
Constraint{S}(confun,          n, m, p, label; inds, term, inputs)
Constraint{S}(confun,          n,    p, label; inds, term, inputs)  # Terminal constraint
```

# Arguments
* `confun`: Inplace constraint function, either `confun!(v, x, u)` or `confun!(v, xN)`.
* `∇confun`: Inplace constraint jacobian function, either `∇confun!(V, x, u)` or `∇confun!(V, xN)`
* `n,m,p`: Number of state, controls, and constraints
* `label`: Symbol that unique identifies the constraint
* `inds`: Specifies the indices of `x` and `u` that are passed into `confun`. Default highly recommended.
* `term`: Where the constraint is applicable. One of `(:all, :stage, :terminal)`. Detected automatically based on defined methods of `confun` but can be manually given.
* `inputs`: Which inputs are actually used. One of `(:xu, :x, :u)`. Should be specified.

"""
struct Constraint{S} <: AbstractConstraint{S}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
    inds::Vector{Vector{Int}}
    type::Symbol
    inputs::Symbol

    function Constraint{S}(c::Function, ∇c::Function, p::Int, label::Symbol,
            inds::Vector{Vector{Int}}, type::Symbol, inputs::Symbol) where S <: ConstraintType
        if type ∉ [:all,:stage,:terminal]
            ArgumentError(string(type) * " is not a supported constraint type")
        end
        new{S}(c,∇c,p,label,inds,type,inputs)
    end
end

"$(TYPEDEF) Create a stage-wise constraint, using ForwardDiff to generate the Jacobian"
function Constraint{S}(c::Function, n::Int, m::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)], term=con_methods(c), inputs=:xu) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c, ∇c, p, label, inds, term, inputs)
end

"Create a terminal constraint using ForwardDiff"
function Constraint{S}(c::Function, n::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)], term=:terminal, inputs=:x) where S<:ConstraintType
    m = 0
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c, ∇c, p, label, inds, term, inputs)
end

"$(TYPEDEF) Create a constraint, providing an analytical Jacobian"
function Constraint{S}(c::Function, ∇c::Function, n::Int, m::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:m)], term=con_methods(c), inputs=:xu) where S<:ConstraintType
    Constraint{S}(c, ∇c, p, label, inds, term, inputs)
end

"Create a terminal constraint with analytical Jacobian"
function Constraint{S}(c::Function, ∇c::Function, n::Int, p::Int, label::Symbol;
        inds=[collect(1:n), collect(1:0)], term=:terminal, inputs=:x) where S<:ConstraintType
    Constraint{S}(c, ∇c, p, label, inds, term, inputs)
end

function Constraint()
    Constraint{Null}(x->nothing, x->nothing, 0, 0, 0, :null, term, :null)
end

evaluate!(v::AbstractVector, con::Constraint, x::AbstractVector, u::AbstractVector) = is_stage(con) ? con.c(v, x, u) : nothing
evaluate!(v::AbstractVector, con::Constraint, x::AbstractVector) = is_terminal(con) ? con.c(v,x) : nothing
jacobian!(V::AbstractMatrix, con::Constraint, x::AbstractVector, u::AbstractVector) = is_stage(con) ? con.∇c(V, x, u) : nothing
jacobian!(V::AbstractMatrix, con::Constraint, x::AbstractVector) = is_terminal(con) ? con.∇c(V,x) : nothing

violation!(v, con::Constraint{Equality}, x, u) = evaluate!(v, con, x, u)
function violation!(v, con::AbstractConstraint{Inequality}, x, u)
    evaluate!(v, con, x, u);
    for i in eachindex(v)
        v[i] = pos(v[i])
    end
end

violation!(v, con::Constraint{Equality}, x) = evaluate!(v, con, x)
function violation!(v, con::AbstractConstraint{Inequality}, x)
    evaluate!(v, con, x);
    for i in eachindex(v)
        v[i] = pos(v[i])
    end
end

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


"""$(TYPEDEF) Linear bound constraint on states and controls
# Constructors
```julia
BoundConstraint(n, m; x_min, x_max, u_min, u_max)
```
Any of the bounds can be ±∞. The bound can also be specifed as a single scalar, which applies the bound to all state/controls.
"""
struct BoundConstraint{T} <: AbstractConstraint{Inequality}
    x_max::Vector{T}
    x_min::Vector{T}
    u_max::Vector{T}
    u_min::Vector{T}
    jac::SparseMatrixCSC{T,Int}
    label::Symbol
    active::NamedTuple{(:x_max, :u_max, :x_min, :u_min, :all, :x_all),NTuple{6,BitArray{1}}}
    inds::Vector{Vector{Int}}
    part::NamedTuple{(:x_max, :u_max, :x_min, :u_min),NTuple{4,UnitRange{Int}}}
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
     jac_bnd = [Diagonal(1.0I,n+m); -Diagonal(1.0I,n+m)]
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
         active = merge(active, (all=active_all, x_all=[active.x_max; falses(m); active.x_min; falses(m)]))
         ∇c_trim(C,x,u) = copyto!(C,jac_bnd[active_all,:])
     else
         active = (x_max=trues(n),u_max=trues(m),
                   x_min=trues(n),u_min=trues(m),
                   all=trues(2(n+m)), x_all=[trues(n); falses(m); trues(m); falses(m)])
     end
     return BoundConstraint(x_max, x_min, u_max, u_min, jac_bnd, :bound, active, inds2, part)
end

"""```julia
combine(::BoundConstraint, ::BoundConstraint)
```
Stack two bound constraints. Useful when the state is augmented.
"""
function combine(bnd1::BoundConstraint, bnd2::BoundConstraint)
    n1,m1 = length(bnd1.x_max), length(bnd1.u_max)
    n2,m2 = length(bnd2.x_max), length(bnd2.u_max)
    bnd = BoundConstraint(n1+n2, m1+m2,
                          x_max=[bnd1.x_max; bnd2.x_max],
                          x_min=[bnd1.x_min; bnd2.x_min],
                          u_max=[bnd1.u_max; bnd2.u_max],
                          u_min=[bnd1.u_min; bnd2.u_min])
end

"""```julia
vcat(bnd1::BoundConstraint, bnd2::BoundConstraint)
```
Stack two bound constraints. Useful when the state is augmented. Equivalent to combine(bnd1, bnd2).
"""
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
    # TODO: avoid concatenation
    copyto!(V, bnd.jac[bnd.active.all,:])
end

function jacobian!(V::AbstractMatrix, bnd::BoundConstraint, x::AbstractVector)
    copyto!(V, bnd.jac[bnd.active.x_all,1:length(x)])
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

"""$(SIGNATURES)
A constraint where x,y positions of the state must remain a distance r from a circle centered at x_obs
Assumes x,y are the first two dimensions of the state vector
"""
function planar_obstacle_constraint(n, m, x_obs, r_obs, label=:obstacle)
    c(v,x,u) = v[1] = circle_constraint(x, x_obs, r_obs)
    c(v,x) = circle_constraint(x, x_obs, r_obs)
    Constraint{Inequality}(c, n, m, 1, label)
end

"""```julia
goal_constraint(xf)
```
Creates a terminal equality constraint specifying the goal. All states must be specified.
"""
function goal_constraint(xf::Vector{T}) where T
    n = length(xf)
    terminal_constraint(v,xN) = copyto!(v,xN-xf)
    terminal_jacobian(C,xN) = copyto!(C,Diagonal(I,n))
    Constraint{Equality}(terminal_constraint, terminal_jacobian, n, :goal, [collect(1:n),collect(1:0)], :terminal, :x)
end

function infeasible_constraints(n::Int, m::Int)
    idx_inf = (n+m) .+ (1:n)
    u_inf = m .+ (1:n)
    ∇inf = zeros(n,2n+m)
    ∇inf[:,idx_inf] = Diagonal(1.0I,n)
    inf_con(v,x,u) = copyto!(v, u)
    inf_jac(C,x,u) = copyto!(C, ∇inf)
    Constraint{Equality}(inf_con, inf_jac, n, :infeasible, [collect(1:n), collect(u_inf)], :stage, :u)
end
