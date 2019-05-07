using Test
import TrajectoryOptimization: bound_constraint, ConstraintSet, StageConstraintSet, TerminalConstraintSet, stage, ProblemConstraints
using PartedArrays
using LinearAlgebra

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
C = zeros(p1,n+m)
con.∇c(C,x,u);
@test con.∇c(x,u) == jacob_c(x,u)
@test C == jacob_c(x,u)


# Custom inequality constraint
p2 = 2
c2(v,x,u) = begin v[1] = sin(x[1]); v[2] = sin(x[3]) end
∇c2(Z,x,u) = begin Z[1,1] = cos(x[1]); Z[2,3] = cos(x[3]); end
con2 = Constraint{Inequality}(c2,∇c2,n,m,p2,:ineq)

# Bound constraint
x_max = [5,5,Inf]
x_min = [-10,-5,0]
u_max = 0
u_min = -10
p3 = 2(n+m)
bnd = bound_constraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max, trim=false)
v = zeros(p3)
bnd.c(v,x,u)
@test v == [-4,-3,-Inf,-5,5,-11,-7,-3,-5,-15]
C = BlockArray(zeros(p3,n+m),create_partition2((p3,),(n,m),(:x,),(:x,:u)))
bnd.∇c(C,x,u)
@test C.xx == [Diagonal(I,n); zeros(m,n); -Diagonal(I,n); zeros(m,n)]
@test C.xu == [zeros(n,m); Diagonal(I,m); zeros(n,m); -Diagonal(I,m)]

# Trimmed bound constraint
bnd = bound_constraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max)
p3 = 2(n+m)-1
v = zeros(p3)
bnd.c(v,x,u)
@test v == [-4,-3,-5,5,-11,-7,-3,-5,-15]

# Create Constraint Set
C = [con,con2,bnd]
@test C isa AbstractConstraintSet
@test C isa StageConstraintSet
@test !(C isa TerminalConstraintSet)

@test findall(C,Inequality) == [false,true,true]
@test split(C) == ([con2,bnd],[con,])
count_constraints(C)
@test count_constraints(C) == (p2+p3,p1)
@test inequalities(C) == [con2,bnd]
@test equalities(C) == [con,]
@test bounds(C) == [bnd,]
@test labels(C) == [:custom,:ineq,:bound]


# Terminal Constraint
cterm(v,x) = begin v[1] = x[1] - 5; v[2] = x[1]*x[2] end
∇cterm(x) = [1 0 0; x[2] x[1] 0]
∇cterm(A,x) = copyto!(A,∇cterm(x))
p_term = 2
v = zeros(p_term)
cterm(v,x)
con_term = TerminalConstraint{Equality}(cterm,∇cterm,n,p_term,:terminal)
v2 = zeros(p_term)
con_term.c(v2,x)
@test v == v2
A = zeros(p_term,n)
@test con_term.∇c(A,x) == ∇cterm(x)

bnd_term = bound_constraint(n,x_max=x_max,x_min=x_min,trim=true)
pI_N = 5
@test length(bnd_term) == pI_N

C_term = [con_term,bnd_term]
C2 = [con,con2,bnd,con_term,bnd_term]
@test C2 isa AbstractConstraintSet
@test C_term isa TerminalConstraintSet

@test terminal(C2) == C_term
@test terminal(C_term) == C_term
@test stage(C2) == C
@test isempty(terminal(C))
@test isempty(stage(C_term))
@test count_constraints(C_term) == (pI_N,p_term)
@test count_constraints(C2) == (p2+p3+pI_N,p1+p_term)
@test split(C2) == ([con2,bnd,bnd_term],[con,con_term])
@test split(C2) == (inequalities(C2),equalities(C2))
@test bounds(C2) == [bnd,bnd_term]
@test labels(C2) == [:custom,:ineq,:bound,:terminal,:terminal_bound]
@test Vector{Constraint}(stage(C2)) isa StageConstraintSet
bounds(C2)

v_stage = BlockVector(stage(C2))
v_term = BlockVector(terminal(C2))
v_stage2 = BlockVector(stage(C2))
v_term2 = BlockVector(terminal(C2))
evaluate!(v_stage,C,x,u)
evaluate!(v_term,C_term,x)
evaluate!(v_stage2,C2,x,u)
evaluate!(v_term2,C2,x)
@test v_stage == v_stage2
@test v_term == v_term2

c_jac = BlockMatrix(C,n,m)
@test size(c_jac) == (p1+p2+p3,n+m)
@test size(c_jac.custom) == (p1,n+m)
@test size(c_jac.x) == (p1+p2+p3,n)
@test size(c_jac.u) == (p1+p2+p3,m)


# Augment State
m_inf = m+n
con_inf = TrajectoryOptimization.infeasible_constraints(n,m)
u_inf = [u; 5; -5; 10]
v = zeros(n)
con_inf.c(v,x,u_inf[con_inf.inds[2]])
@test v == [5,-5,10]

C_inf = [con,con2,bnd,con_inf]
v_stage = BlockVector(stage(C))
v_inf = BlockVector(stage(C_inf))
TrajectoryOptimization.evaluate!(v_stage,C,x,u_inf)
TrajectoryOptimization.evaluate!(v_inf,C_inf,x,u_inf)
@test v_inf == [v_stage;5;-5;10]

BlockVector(Int64,C_term)
# Test constrained cost stuff
N = 21
p = num_constraints(C)
c_part = create_partition(C)
c_part2 = create_partition2(C,n,m)
λ = [BlockArray(zeros(p),c_part) for k = 1:N-1]
μ = [BlockArray(ones(p),c_part) for k = 1:N-1]
a = [BlockArray(ones(Bool,p),c_part) for k = 1:N-1]
cval = [BlockArray(zeros(p),c_part) for k = 1:N-1]
∇cval = [BlockArray(zeros(p,n+m),c_part2) for k = 1:N-1]
Iμ = [BlockArray(zeros(p),c_part) for k = 1:N-1]
push!(λ,BlockVector(C_term))
push!(μ,BlockVector(C_term))
push!(a,BlockVector(Bool,C_term))
push!(cval,BlockVector(C_term))
push!(∇cval,BlockMatrix(C_term,n,m))
push!(Iμ,BlockVector(C_term))
X = to_dvecs(rand(n,N))
U = to_dvecs(rand(m,N-1))
TrajectoryOptimization.update_constraints!(cval,ProblemConstraints(C2,N),X,U)
v = zeros(p1)
c(v,X[5],U[5])
@test cval[5].custom == v
