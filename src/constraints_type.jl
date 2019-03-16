using BlockArrays, Test, ForwardDiff
using BenchmarkTools

abstract type ConstraintType end
# abstract type StageConstraint <: ConstraintType end
# abstract type TerminalConstraint <: ConstraintType end
abstract type Equality <: ConstraintType end
abstract type Inequality <: ConstraintType end

abstract type AbstractConstraint{S<:ConstraintType} end
struct Constraint{S} <: AbstractConstraint{S}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
end
function Constraint{S}(c::Function,n::Int,m::Int,p::Int,label::Symbol) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c,∇c,p,label)
end

struct TerminalConstraint{S} <: AbstractConstraint{S}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
end
function TerminalConstraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,p)
    TerminalConstraint{S}(c,∇c,p,label)
end
Constraint{S}(c::Function,n::Int,p::Int,label::Symbol) where S<:ConstraintType = TerminalConstraint(c,n,p,label)


type(::AbstractConstraint{S}) where S = S
length(C::AbstractConstraint) = C.p

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
         ∇c_trim(A,B,v,x,u) = begin copyto!(A,jac_A[active_all,:]); copyto!(B,jac_B[active_all,:]); bound_trim(v,x,u) end
         Constraint{Inequality}(bound_trim,∇c_trim,count(active_all),:bound)
     else
         # Generate function
         function bound_all(v,x,u)
             v[inds.x_max] = x - x_max
             v[inds.u_max] = u - u_max
             v[inds.x_min] = x_min - x
             v[inds.u_min] = u_min - u
         end
         ∇c(A,B,v,x,u) = begin copyto!(A,jac_A); copyto!(B,jac_B); bound_all(v,x,u); return jac_A, jac_B end
         Constraint{Inequality}(bound_all,∇c,2(n+m),:bound)
     end
 end

function generate_jacobian(f!::Function,n::Int,m::Int,p::Int=n)
    inds = (x=1:n,u=n .+ (1:m), xx=(1:n,1:n),xu=(1:n,n .+ (1:m)))
    Z = zeros(p,n+m)
    z = zeros(n+m)
    f_aug(dZ::AbstractVector,Z::AbstractVector) = f!(dZ,view(Z,inds.x), view(Z,inds.u))
    ∇f!(Z,v,z) = ForwardDiff.jacobian!(Z,f_aug,v,z)
    ∇f!(A,B,v,x,u) = begin
        z[inds.x] = x
        z[inds.u] = u
        ∇f!(Z,v,z)
        copyto!(A,Z[inds.xx...])
        copyto!(B,Z[inds.xu...])
    end
    return ∇f!, f_aug
end

function generate_jacobian(f!::Function,n::Int,p::Int=n)
    ∇f!(A,v,x) = ForwardDiff.jacobian!(A,f!,v,x)
    return ∇f!, f!
end

# Constraint Sets
ConstraintSet = Vector{<:AbstractConstraint{S} where S}
StageConstraintSet = Vector{T} where T<:Constraint
TerminalConstraintSet = Vector{T} where T<:TerminalConstraint

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

function calculate!(c::BlockVector, C::StageConstraintSet, x, u)
    for con in C
        con.c(c[con.label],x,u)
    end
end
calculate!(c::BlockVector, C::ConstraintSet, x, u) = calculate!(c,stage(C),x,u)
function calculate!(c::BlockVector, C::TerminalConstraintSet, x)
    for con in C
        con.c(c[con.label],x)
    end
end
calculate!(c::BlockVector, C::ConstraintSet, x) = calculate!(c,terminal(C),x)

labels(C::ConstraintSet) = [c.label for c in C]
terminal(C::ConstraintSet) = Vector{TerminalConstraint}(filter(x->isa(x,TerminalConstraint),C))
stage(C::ConstraintSet) = Vector{Constraint}(filter(x->isa(x,Constraint),C))
inequalities(C::ConstraintSet) = filter(x->isa(x,AbstractConstraint{Inequality}),C)
equalities(C::ConstraintSet) = filter(x->isa(x,AbstractConstraint{Equality}),C)
bounds(C::ConstraintSet) = filter(x->x.label==:bound,C)
Base.findall(C::ConstraintSet,T::Type) = isa.(C,Constraint{T})
BlockArrays.create_partition(C::ConstraintSet) = create_partition(Tuple(length.(C)),Tuple(labels(C)))
BlockArrays.BlockArray(C::ConstraintSet) = BlockArray(zeros(sum(length.(C))), create_partition(C))

n,m = 3,2

# Custom Equality Constraint
p1 = 3
c(v,x,u) = begin v[1]  = x[1]^2 + x[2]^2 - 5; v[2:3] =  u - ones(2,1) end
jacob_c(x,u) = [2x[1] 2x[2] 0 0 0;
                0     0     0 1 0;
                0     0     0 0 1];
v = zeros(p1)
x = [1,2,3]
u = [-5,5]
c(v,x,u)
@test v == [0,-6,4]

# Test constraint function
con = Constraint{Equality}(c,n,m,p1,:custom)
con.c(v,x,u)
@test v == [0,-6,4]

# Test constraint jacobian
A = zeros(p1,n)
B = zeros(p1,m)
C = zeros(p1,n+m)
con.∇c(A,B,v,x,u);
@test A == jacob_c(x,u)[:,1:n]
@test B == jacob_c(x,u)[:,n+1:end]

# Joint jacobian function
con.∇c(C,v,[x;u])
@test C == jacob_c(x,u)


# Custom inequality constraint
p2 = 2
c2(v,x,u) = begin v[1] = sin(x[1]); v[2] = sin(x[3]) end
∇c2(A,B,v,x,u) = begin A[1,1] = cos(x[1]); A[2,3] = cos(x[3]); c2(v,x,u) end
con2 = Constraint{Inequality}(c2,∇c2,p2,:ineq)

# Bound constraint
x_max = [5,5,Inf]
x_min = [-10,-5,0]
u_max = 0
u_min = -10
p3 = 2(n+m)
bnd = bound_constraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max)
v = zeros(p3)
bnd.c(v,x,u)
@test v == [-4,-3,-Inf,-5,5,-11,-7,-3,-5,-15]
A = zeros(p3,n)
B = zeros(p3,m)
C = zeros(p3,n+m)
bnd.∇c(A,B,v,x,u)
@test A == [Diagonal(I,n); zeros(m,n); -Diagonal(I,n); zeros(m,n)]
@test B == [zeros(n,m); Diagonal(I,m); zeros(n,m); -Diagonal(I,m)]

# Trimmed bound constraint
bnd = bound_constraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max,trim=true)
p3 = 2(n+m)-1
v = zeros(p3)
bnd.c(v,x,u)
@test v == [-4,-3,-5,5,-11,-7,-3,-5,-15]

# Create Constraint Set
C = [con,con2,bnd]
@test C isa ConstraintSet
@test C isa StageConstraintSet
@test !(C isa TerminalConstraintSet)

@test findall(C,Inequality) == [false,true,true]
@test split(C) == ([con2,bnd],[con,])
@test count_constraints(C) == (p2+p3,p1)
@test inequalities(C) == [con2,bnd]
@test equalities(C) == [con,]
@test bounds(C) == [bnd,]
@test labels(C) == [:custom,:ineq,:bound]

c_part = create_partition(C)
cs2 = BlockArray(C)
calculate!(cs2,C,x,u)

obj = LQRObjective(Diagonal(I,n),Diagonal(I,m),Diagonal(I,n),2.,zeros(n),ones(n))
obj = ConstrainedObjective(obj,u_max=u_max,u_min=u_min,x_max=x_max,x_min=x_min,cI=c2,cE=c)
cfun, = generate_constraint_functions(obj)
cs = zeros(obj.p)
cfun(cs,x,u)

# Test sum since they're ordered differently
@test sum(cs) == sum(cs2)

@btime calculate!(cs2,C,x,u)
@btime cfun(cs,x,u)


# Terminal Constraint
cterm(v,x) = begin v[1] = x[1] - 5; v[2] = x[1]*x[2] end
∇cterm(x) = [1 0 0; x[2] x[1] 0]
∇cterm(A,x) = copyto!(A,∇cterm(x))
p_N = 2
v = zeros(p_N)
cterm(v,x)
con_term = TerminalConstraint{Equality}(cterm,∇cterm,p_N,:terminal)
v2 = zeros(p_N)
con_term.c(v2,x)
@test v == v2
A = zeros(p_N,n)
con_term.∇c(A,x) == ∇cterm(x)

C_term = [con_term,]
C2 = [con,con2,bnd,con_term]
@test C2 isa ConstraintSet
@test C_term isa TerminalConstraintSet

@test terminal(C2) == C_term
@test terminal(C_term) == C_term
@test stage(C2) == C
@test isempty(terminal(C))
@test isempty(stage(C_term))
@test count_constraints(C_term) == (0,p_N)
@test count_constraints(C2) == (p2+p3,p1+p_N)
@test split(C2) == ([con2,bnd],[con,con_term])
@test split(C2) == (inequalities(C2),equalities(C2))
@test bounds(C2) == [bnd,]
@test labels(C2) == [:custom,:ineq,:bound,:terminal]
terminal(C2)
Vector{Constraint}(stage(C2)) isa StageConstraintSet

v_stage = BlockArray(stage(C2))
v_term = BlockArray(terminal(C2))
v_stage2 = BlockArray(stage(C2))
v_term2 = BlockArray(terminal(C2))
calculate!(v_stage,C,x,u)
calculate!(v_term,C_term,x)
calculate!(v_stage2,C2,x,u)
calculate!(v_term2,C2,x)
@test v_stage == v_stage2
@test v_term == v_term2
