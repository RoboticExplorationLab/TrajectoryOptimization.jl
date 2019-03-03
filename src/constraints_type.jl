using BlockArrays

abstract type ConstraintType end
abstract type Equality <: ConstraintType end
abstract type Inequality <: ConstraintType end
ConstraintSet = Vector{Constraint}

struct Constraint{S<:ConstraintType}
    c::Function
    ∇c::Function
    p::Int
    label::Symbol
end

function Constraint{S}(c::Function,n::Int,m::Int,p::Int,label::Symbol) where S<:ConstraintType
    ∇c,c_aug = generate_jacobian(c,n,m,p)
    Constraint{S}(c,∇c,p,label)
end
type(::Constraint{S}) where S = S
length(C::Constraint) = C.p

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
         ∇c(A,B,v,x,u) = begin copyto!(A,jac_A); copyto!(B,jac_B); bound_all(v,x,u) end
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

function count_constraints(C::ConstraintSet)
    pI = 0
    pE = 0
    for c in C
        if c isa Constraint{Equality}
            pE += c.p
        elseif c isa Constraint{Inequality}
            pI += c.p
        end
    end
    return pI,pE
end

function Base.split(C::ConstraintSet)
    E = Constraint{Equality}[]
    I = Constraint{Inequality}[]
    for c in C
        if c isa Constraint{Equality}
            push!(E,c)
        elseif c isa Constraint{Inequality}
            push!(I,c)
        end
    end
    return E,I
end

labels(C::ConstraintSet) = [c.label for c in C]

function calculate!(c::BlockVector, C::ConstraintSet, x, u)
    for con in C
        con.c(c[con.label],x,u)
    end
end


inequalities(C::ConstraintSet) = filter(x->isa(x,Constraint{Inequality}),C)
equalities(C::ConstraintSet) = filter(x->isa(x,Constraint{Equality}),C)
bounds(C::ConstraintSet) = filter(x->x.label==:bound,C)
Base.findall(C::ConstraintSet,T::Type) = isa.(C,Constraint{T})
findall(C,Inequality)
split(C)
count_constraints(C)
inequalities(C)
equalities(C)
bounds(C)
labels(C)
p = sum(length.(C))
cs = zeros(p)
c_part = create_partition(Tuple(length.(C)),Tuple(labels(C)))
cs2 = BlockArray(cs,c_part)
calculate!(cs2,C,x,u)

obj = LQRObjective(Diagonal(I,n),Diagonal(I,m),Diagonal(I,n),2.,zeros(n),ones(n))
obj = ConstrainedObjective(obj,u_max=u_max,u_min=u_min,x_max=x_max,x_min=x_min,cI=c2,cE=c)
cfun, = generate_constraint_functions(obj)
cfun(cs,x,u)
cs == cs2

@btime calculate!(cs2,C,x,u)
@btime cfun(cs,x,u)

using BlockArrays


n,m,p = 3,2,3
c(v,x,u) = begin v[1]  = x[1]^2 + x[2]^2 - 5; v[2:3] =  u - ones(2,1) end
v = zeros(p)
x = [1,2,3]
u = [-5,5]
c(v,x,u)
v == [0,-6,4]

jacob_c(x,u) = [2x[1] 2x[2] 0 0 0;
                0     0     0 1 0;
                0     0     0 0 1];

con = Constraint{Equality}(c,n,m,p,:custom)
v = zero(v)
con.c(v,x,u)
v == [0,-6,4]
A = zeros(p,n)
B = zeros(p,m)
C = zeros(p,n+m)
con.∇c(A,B,v,x,u);
A == jacob_c(x,u)[:,1:n]
B == jacob_c(x,u)[:,n+1:end]
con.∇c(C,v,[x;u])
C == jacob_c(x,u)

p2 = 2
c2(v,x,u) = begin v[1] = sin(x[1]); v[2] = sin(x[3]) end
∇c2(A,B,v,x,u) = begin A[1,1] = cos(x[1]); A[2,3] = cos(x[3]); c2(v,x,u) end
con2 = Constraint{Inequality}(c2,∇c2,p2,:ineq)

x_max = [5,5,5]
x_min = [-10,-5,0]
u_max = 0
u_min = -10
p = 2(n+m)
bnd = bound_constraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max)
v = zeros(p)
bnd.c(v,x,u)
v == [-4,-3,-Inf,-5,5,-11,-7,-3,-5,-15]
A = zeros(p,n)
B = zeros(p,m)
C = zeros(p,n+m)
bnd.∇c(A,B,v,x,u)

C = [con,con2,bnd]

bnd = bound_constraint(n,m,x_max=x_max,x_min=x_min,u_min=-10,u_max=0,trim=true)
bnd.p == 9
p = 9
v = zeros(p)
bnd.c(v,x,u)
v == [-4,-3,-5,5,-11,-7,-3,-5,-15]
A = zeros(p,n)
B = zeros(p,m)
C = zeros(p,n+m)
bnd.∇c(A,B,v,x,u)
