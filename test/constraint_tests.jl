using Test
# # Test constraint stuff
# n,m = 3,2
# cE(x,u) = [2x[1:2]+u;
#           x'x + 5]
# pE = 3
# cE(x) = [cos(x[1]) + x[2]*x[3]; x[1]*x[2]^2]
# pE_N = 2
# cI(x,u) = [x[3]-x[2]; u[1]*x[1]]
# pI = 2
# pI_N = 0
#
# model, obj = Dynamics.dubinscar
# count_inplace_output(cI,n,m)
# obj_con = ConstrainedObjective(obj,cE=cE,cI=cI)
# @test obj_con.p == pE + pI
# @test obj_con.pI == pI
# @test obj_con.pI_N == pI_N
# @test obj_con.p_N == n + pE_N + pI_N
#
# N = 5
# solver = Solver(model, obj, N=N)
# @test solver.state.constrained == true
# @test get_num_constraints(solver) == (5,2,3)
# @test original_constraint_inds(solver) == trues(5)
# @test get_constraint_labels(solver) == ["custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality"]
#
# # Add state and control bounds
# obj_con = update_objective(obj_con, u_min=[-10,-Inf], u_max=10, x_min=[-Inf,-10,-Inf], x_max=[10,12,10])
# pI_bnd = 1 + m + 1 + n
# @test obj_con.pI == pI + pI_bnd
# @test obj_con.p == pI + pI_bnd + pE
# p = obj_con.p
# pI = obj_con.pI
# pE = p-pI
#
# N = 5
# solver = Solver(model, obj_con, N=N)
# @test solver.state.constrained == true
# @test get_num_constraints(solver) == (5+pI_bnd,2+pI_bnd,3)
# @test original_constraint_inds(solver) == trues(5+pI_bnd)
# @test get_constraint_labels(solver) == ["control (upper bound)", "control (upper bound)", "control (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
#     "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality"]
#
# # Infeasible controls
# solver_inf = Solver(model, obj_con, N=N)
# solver_inf.opts.infeasible = true
# @test solver_inf.opts.constrained == true
# @test get_num_constraints(solver_inf) == (p+n,pI,pE+n)
# @test original_constraint_inds(solver_inf) == [trues(p); falses(n)]
# @test get_constraint_labels(solver_inf) == ["control (upper bound)", "control (upper bound)", "control (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
#     "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality",
#     "* infeasible control","* infeasible control","* infeasible control"]
#
# # Minimum time
# obj_mintime = update_objective(obj_con, tf=:min)
# solver_min = Solver(model, obj_mintime, N=N)
# @test solver_min.opts.constrained == true
# @test get_num_constraints(solver_min) == (p+3,pI+2,pE+1)
# @test original_constraint_inds(solver_min) == [true; true; false; true; false; trues(4); trues(5); false]
# @test get_constraint_labels(solver_min) == ["control (upper bound)", "control (upper bound)", "* √dt (upper bound)", "control (lower bound)", "* √dt (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
#     "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality",
#     "* √dt (equality)"]
#
# # Minimum time and infeasible
# obj_mintime = update_objective(obj_con, tf=:min)
# solver_min = Solver(model, obj_mintime, N=N)
# solver_min.opts.infeasible = true
# @test solver_min.opts.constrained == true
# @test get_num_constraints(solver_min) == (p+3+n,pI+2,pE+1+n)
# @test original_constraint_inds(solver_min) == [true; true; false; true; false; trues(4); trues(5); falses(4)]
# @test get_constraint_labels(solver_min) == ["control (upper bound)", "control (upper bound)", "* √dt (upper bound)", "control (lower bound)", "* √dt (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
#     "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality",
#     "* infeasible control","* infeasible control","* infeasible control","* √dt (equality)"]
# get_constraint_labels(solver_min)
#


####################
# NEW CONSTRAINTS #
###################

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
C = BlockArray(zeros(p3,n+m),create_partition2((p3,),(n,m),(:x,),(:x,:u)))
bnd.∇c(C,x,u)
@test C.xx == [Diagonal(I,n); zeros(m,n); -Diagonal(I,n); zeros(m,n)]
@test C.xu == [zeros(n,m); Diagonal(I,m); zeros(n,m); -Diagonal(I,m)]

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


# Terminal Constraint
cterm(v,x) = begin v[1] = x[1] - 5; v[2] = x[1]*x[2] end
∇cterm(x) = [1 0 0; x[2] x[1] 0]
∇cterm(A,x) = copyto!(A,∇cterm(x))
p_term = 2
v = zeros(p_term)
cterm(v,x)
con_term = TerminalConstraint{Equality}(cterm,∇cterm,p_term,:terminal)
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
@test C2 isa ConstraintSet
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
update_constraints!(cval,C2,X,U)
v = zeros(p1)
c(v,X[5],U[5])
@test cval[5].custom == v

Q = Diagonal(1I,n)
R = Diagonal(1I,m)
Qf = Diagonal(10I,n)
xf = rand(n)
quadcost = LQRCost(Q,R,Qf,xf)

alcost = AugmentedLagrangianCost(quadcost,C2,λ,μ,a,cval,∇cval)
stage_cost(alcost,X[1],U[1],1)


obj = UnconstrainedObjective(quadcost,2.0,zeros(n),ones(n))
obj = ConstrainedObjective(obj,u_max=u_max,u_min=u_min,x_max=x_max,x_min=x_min,
    cI=c2,cE=c,use_xf_equality_constraint=false,cE_N=cterm)
solver = Solver(Dynamics.dubinscar[1],obj,N=N)
dt = solver.dt

# Test Constraint evaluations
cs = zeros(obj.p)
cs2 = BlockVector(C)
solver.c_fun(cs,x,u)
evaluate!(cs2,C,x,u)
@test sum(cs) ≈ sum(cs2)  # Test sum since they're ordered differently

@btime solver.c_fun($cs,$x,$u)
@btime evaluate!($cs2,$C,$x,$u)

# Test constraint jacobians
cx,cu = zeros(obj.p,n), zeros(obj.p,m)
cz = BlockMatrix(C,n,m)
solver.c_jacobian(cx,cu,x,u)
jacobian!(cz,C,x,u)
@test sum(cz) ≈ sum(cx) + sum(cu)

@btime solver.c_jacobian($cx,$cu,$x,$u)
@btime jacobian!($cz,$C,$x,$u)

num_constraints(C)
get_num_constraints(solver)
res = init_results(solver,Matrix{Float64}(undef,0,0),to_array(U),λ=λ,μ=μ)
copyto!(res.X,X)
update_constraints!(res,solver)
@test cost(alcost,X,U,dt) - cost(quadcost,X,U,dt) ≈ cost_constraints(solver,res)
@test _cost(solver,res) == cost(quadcost,X,U,dt)
@test cost(alcost,X,U,dt) ≈ cost(solver,res)

function solver_cost(solver::Solver,res)
    update_constraints!(res,solver)
    cost(solver,res)
end
solver_cost(solver,res)
@btime solver_cost($solver,$res)
@btime cost($alcost,$X,$U,$dt)

@btime _cost($solver,$res)
@btime cost($quadcost,$X,$U,$dt)

using InteractiveUtils
@code_warntype penalty_cost(cval[1],λ[1],μ[1])
@code_warntype stage_constraint_cost(alcost,X[1],U[1],1)

@btime constraint_cost($cval,$λ,$μ,$a,$Iμ)
@btime constraint_cost($cval,$λ,$μ,$a)
# @btime active_set!($a,$v_stage,$λ,$C)
@code_warntype inequality_active!(a,v_stage,λ)
using InteractiveUtils

taylor_expansion(alcost,X[1],U[1],1)
taylor_expansion(alcost,X[N])
a.equality .= true
