using Test
using ForwardDiff
import TrajectoryOptimization: ConstraintSet, stage, ProblemConstraints, BoundConstraint
using PartedArrays
using LinearAlgebra

n,m = 3,2

# Custom Equality Constraint
p1 = 3
c(v,x,u) = begin v[1]  = x[1]^2 + x[2]^2 - 5; v[2:3] =  u - ones(2,1) end
c(v,x) = c(v,x,zeros(2))
jacob_c(x,u) = [2x[1] 2x[2] 0 0 0;
                0     0     0 1 0;
                0     0     0 0 1];
jacob_c(x) = [2x[1] 2x[2] 0;
                0     0     0;
                0     0     0];

v = zeros(p1)
x = [1.,2,3]
u = [-5.,5]
c(v,x,u)
@test v == [0,-6,4]

c(v,x)
@test v == [0, -1, -1]

J = zeros(p1,n+m)
J_term = zeros(p1,n)
ForwardDiff.jacobian!(J_term,c,v,x)
@test J_term == jacob_c(x)

import TrajectoryOptimization: generate_jacobian
∇c, = generate_jacobian(c, n, m, p1)
methods(∇c)
∇c(J,x,u)
@test J == jacob_c(x,u)
@test ∇c(J_term,x) == jacob_c(x)

# Test constraint function
con = Constraint{Equality}(c,n,m,p1,:custom)
evaluate!(v, con, x, u)
@test v == [0,-6,4]
evaluate!(v, con, x)
@test v == [0, -1, -1]
@test length(con) == p1

# Test constraint jacobian
con.∇c(J,x,u);
jacobian!(J, con, x, u)
@test J == jacob_c(x,u)
jacobian!(J_term, con, x)
@test J_term == jacob_c(x)


# Custom inequality constraint
p2 = 2
c2(v,x,u) = begin v[1] = sin(x[1]); v[2] = sin(x[3]) end
∇c2(Z,x,u) = begin Z[1,1] = cos(x[1]); Z[2,3] = cos(x[3]); end
con2 = Constraint{Inequality}(c2,∇c2,n,m,p2,:ineq)


# Bound constraint
x_max = [5,5,Inf]
x_min = [-10,-5,0.]
u_max = 0.
u_min = -10.

p3 = 2(n+m)
bnd = BoundConstraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max, trim=false)
v = zeros(p3)
evaluate!(v, bnd, x, u)
@test v == [-4,-3,-Inf,-5,5,-11,-7,-3,-5,-15]
C = PartedArray(zeros(p3,n+m),create_partition2((p3,),(n,m),Val((:xx,:xu))))
jacobian!(C, bnd, x, u)
@test C.xx == [Diagonal(1I,n); zeros(m,n); -Diagonal(1I,n); zeros(m,n)]
@test C.xu == [zeros(n,m); Diagonal(1I,m); zeros(n,m); -Diagonal(1I,m)]
@test length(bnd) == p3

v = zeros(2n)
evaluate!(v, bnd, x)
@test v == [-4,-3,-Inf,-11,-7,-3]
C = zeros(2n, n)
@test jacobian!(C, bnd, x) == [Diagonal(1I,n); -Diagonal(1I,n)]
@test length(bnd,:terminal) == 2n


# Trimmed bound constraint
bnd = BoundConstraint(n,m,x_max=x_max,x_min=x_min,u_min=u_min,u_max=u_max)
p3 = 2(n+m)-1
p3_N = 2n-1
v = zeros(p3)
evaluate!(v, bnd, x, u)
@test v == [-4,-3,-5,5,-11,-7,-3,-5,-15]
C = zeros(p3, n+m)
jac_bnd = [Diagonal(1I,n+m)[[true;true;false;true;true],:]; -Diagonal(1I,n+m)]
@test jacobian!(C, bnd, x, u) == jac_bnd
@test length(bnd,:stage) == p3

v = zeros(p3_N)
evaluate!(v, bnd, x)
@test v == [-4,-3,-11,-7,-3]
C = zeros(p3_N, n)
jacobian!(C, bnd, x)
jac_bnd_term = [Diagonal(1I,n)[[true;true;false],:]; -Diagonal(1I,n)]
@test C == jac_bnd_term

# Create Constraint Set
C = [con,con2,bnd]
@test C isa ConstraintSet
C = con + con2 + bnd
@test C isa ConstraintSet

@test findall(C,Inequality) == [false,true,true]
@test split(C) == ([con2,bnd],[con,])
@test count_constraints(C) == (p2+p3,p1)
@test inequalities(C) == [con2,bnd]
@test equalities(C) == [con,]
@test bounds(C) == [bnd,]
@test labels(C) == [:custom,:ineq,:bound]
@test TrajectoryOptimization.is_terminal.(C) == [true,false,true]

# Terminal Constraint
cterm(v,x) = begin v[1] = x[1] - 5; v[2] = x[1]*x[2] end
∇cterm(x) = [1 0 0; x[2] x[1] 0]
∇cterm(A,x) = copyto!(A,∇cterm(x))
p_term = 2
v = zeros(p_term)
cterm(v,x)
con_term = Constraint{Equality}(cterm,∇cterm,n,p_term,:terminal)
v2 = zeros(p_term)
con_term.c(v2,x)
@test v == v2
A = zeros(p_term,n)
@test con_term.∇c(A,x) == ∇cterm(x)

C_term = [con,bnd,con_term]
C2 = [con,con2,bnd,con_term]
@test C2 isa ConstraintSet
@test terminal(C2) == C_term
@test terminal(C_term) == C_term
@test stage(C2) == C
count_constraints(C2)
@test split(C2) == ([con2,bnd],[con,con_term])
@test split(C2) == (inequalities(C2),equalities(C2))
@test bounds(C2) == [bnd]
@test labels(C2) == [:custom,:ineq,:bound,:terminal]
@test TrajectoryOptimization.num_stage_constraints(C2) == p1+p2+p3
@test TrajectoryOptimization.num_terminal_constraints(C2) == p3_N + p_term + p1

import TrajectoryOptimization: num_stage_constraints, num_terminal_constraints
num_stage_constraints(C2)
num_constraints(C2,:stage)


v_stage = PartedVector(C2)
v_term = PartedVector(C2,:terminal)
v_stage2 = PartedVector(C2,:stage)
v_term2 = PartedVector(C2,:terminal)
evaluate!(v_stage,C,x,u)
evaluate!(v_term,C_term,x)
evaluate!(v_stage2,C2,x,u)
evaluate!(v_term2,C2,x)
@test v_stage == v_stage2
@test v_term == v_term2

c_jac = PartedMatrix(C,n,m)
@test size(c_jac) == (p1+p2+p3,n+m)
@test size(c_jac.custom) == (p1,n+m)
@test size(c_jac.x) == (p1+p2+p3,n)
@test size(c_jac.u) == (p1+p2+p3,m)
jacobian!(c_jac, C, x, u)
@test c_jac.custom == jacob_c(x,u)
@test c_jac.bound == jac_bnd

c_jac = PartedMatrix(C2,n,m,:terminal)
@test size(c_jac) == (p1+p3_N+p_term,n)
@test size(c_jac.custom) == (p1,n)
@test size(c_jac.ineq) == (0,n)
@test size(c_jac.u) == (p1+p3_N+p_term,0)
@test size(c_jac.terminal) == (p_term,n)
jacobian!(c_jac, C2, x)
jacobian!(c_jac.custom, con, x)
@test c_jac.custom == jacob_c(x)
@test c_jac.bound == jac_bnd_term

# Augment State
m_inf = m+n
con_inf = TrajectoryOptimization.infeasible_constraints(n,m)
u_inf = [u; 5; -5; 10]
v = zeros(n)
con_inf.c(v,x,u_inf[con_inf.inds[2]])
@test v == [5,-5,10]

C_inf = [con,con2,bnd,con_inf]
v_stage = PartedVector(stage(C))
v_inf = PartedVector(stage(C_inf))
TrajectoryOptimization.evaluate!(v_stage,C,x,u_inf)
TrajectoryOptimization.evaluate!(v_inf,C_inf,x,u_inf)
@test v_inf == [v_stage;5;-5;10]

C2
C3 = copy(C2)
@test TrajectoryOptimization.remove_bounds!(C3) == [bnd]
@test C3 == [con,con2,con_term]
@test pop!(C3,:ineq) == con2
@test C3 == [con,con_term]


# ProblemConstraints
CS = [[con], con+con2+bnd, con+con2+bnd, con+bnd]
N = length(CS)
@test CS isa Vector{<:ConstraintSet}
PC = ProblemConstraints(CS)
PC[1] += bnd
@test PC[1] == [con,bnd]
@test num_constraints(PC) == [p1+p3, p1+p2+p3, p1+p2+p3, p1+p3_N]
PC[N] += con_term
@test num_constraints(PC) == [p1+p3, p1+p2+p3, p1+p2+p3, p1+p3_N+p_term]
PC2 = copy(PC)
pop!(PC2[2],:ineq)
@test num_constraints(PC2) == [p1+p3, p1+p3, p1+p2+p3, p1+p3_N+p_term]
@test num_constraints(PC) == [p1+p3, p1+p2+p3, p1+p2+p3, p1+p3_N+p_term]

#TODO create problem and retest this
# Cval, ∇Cval = TrajectoryOptimization.init_constraint_trajectories(PC,n,m,N)
# @test length.(Cval) == num_constraints(PC)
# X = [float.(x) for k = 1:N]
# U = [float.(u) for k = 1:N-1]
# TrajectoryOptimization.update_constraints!(Cval, PC, X, U)
# v = zeros(p1)
# c(v,x,u)
# @test Cval[1].custom == v
# c(v,x)
# @test Cval[N].custom == v
#
# jacobian!(∇Cval, PC, X, U)
# @test ∇Cval[N] == c_jac
