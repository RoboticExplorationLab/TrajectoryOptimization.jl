abstract type AbstractProblem end

"$(TYPEDEF) Trajectory Optimization Problem"
struct Problem <: AbstractProblem
    model::Model
    cost::CostFunction
    constraints::ConstraintSet
    X::Trajectory
    U::Trajectory
    N::Int64
    dt::Float64
end

Problem(model::Model,cost::CostFunction) = Problem(model,cost,
    AbstractConstraint[],[[]],[[]],0,0.0)
Problem(model::Model,cost::CostFunction,U::Trajectory) = Problem(model,
    cost,AbstractConstraint[],[[]],U,length(U)+1,0.0)
Problem(model::Model,cost::CostFunction,U::Trajectory,N::Int,dt::Float64) = Problem(model,
    cost,AbstractConstraint[],[[]],U,N,dt)
Problem(model::Model,cost::CostFunction,X::Trajectory,U::Trajectory) = Problem(
    model,cost,AbstractConstraint[],X,U,length(X),0.0)
Problem(model::Model,cost::CostFunction,U::Array) = Problem(model,cost,
    AbstractConstraint[],[[]],[U[:,k] for k = 1:length(U)],length(U)+1,0.0)
Problem(model::Model,cost::CostFunction,U::Array,N::Int,dt::Float64) = Problem(model,cost,
    AbstractConstraint[],[[]],[U[:,k] for k = 1:length(U)],N,dt)
Problem(model::Model,cost::CostFunction,X::Array,U::Array) = Problem(
    model,cost,AbstractConstraint[],[X[:,k] for k = 1:length(X)],
    [U[:,k] for k = 1:length(U)],length(X),0.0)

function update_problem(p::Problem;
    model=p.model,cost=p.cost,constraints=p.constraints,X=p.X,U=p.U,
    N=p.N,dt=p.dt)

    Problem(model,cost,constraints,X,U,N,dt)
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

f = Dynamics.dubins_dynamics!
n,m = 3,2
model = Model(f,n,m)
x0 = [0; 0.; 0.];
xf = [1; 0.; 0.]; # (ie, swing up)
u0 = [1.;1.]
Q = 1e-3*Diagonal(I,n)
Qf = 100. *Diagonal(I,n)
R = 1e-3*Diagonal(I,m)
tf = 5.
costfun = LQRCost(Q,R,Qf,xf)
cc = copy(costfun)
check_problem(p1)
p1 = Problem(model,costfun)
p2 = Problem(model,costfun,[rand(m) for k = 1:10])
p3 = Problem(model,costfun,[rand(n) for k = 1:11],[rand(m) for k = 1:10])
p4 = update_problem(p3,dt=0.1)
add_constraints!(p3,con)
p3
add_constraints!(p4,con)
p4
aa = [p4.constraints...,C...]

n,m = 3,2
aa = [[]]
aa == [[]]
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
con
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
[C...,con]
[C...,C...]
[C,C]
@test C isa ConstraintSet
append!(C,C)
C
