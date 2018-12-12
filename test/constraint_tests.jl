using Test
# Test constraint stuff
n,m = 3,2
cE(x,u) = [2x[1:2]+u;
          x'x + 5]
pE = 3
cE(x) = [cos(x[1]) + x[2]*x[3]; x[1]*x[2]^2]
pE_N = 2
cI(x,u) = [x[3]-x[2]; u[1]*x[1]]
pI = 2
pI_N = 0

model, obj = Dynamics.dubinscar
obj.tf = 3
count_inplace_output(cI,n,m)
obj_con = ConstrainedObjective(obj,cE=cE,cI=cI)
@test obj_con.p == pE + pI
@test obj_con.pI == pI
@test obj_con.pI_N == pI_N
@test obj_con.p_N == n + pE_N + pI_N

N = 5
solver = Solver(model, obj, N=N)
@test solver.opts.constrained == true
@test get_num_constraints(solver) == (5,2,3)
@test original_constraint_inds(solver) == trues(5)
@test get_constraint_labels(solver) == ["custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality"]

# Add state and control bounds
obj_con = update_objective(obj_con, u_min=[-10,-Inf], u_max=10, x_min=[-Inf,-10,-Inf], x_max=[10,12,10])
pI_bnd = 1 + m + 1 + n
@test obj_con.pI == pI + pI_bnd
@test obj_con.p == pI + pI_bnd + pE
p = obj_con.p
pI = obj_con.pI
pE = p-pI

N = 5
solver = Solver(model, obj_con, N=N)
@test solver.opts.constrained == true
@test get_num_constraints(solver) == (5+pI_bnd,2+pI_bnd,3)
@test original_constraint_inds(solver) == trues(5+pI_bnd)
@test get_constraint_labels(solver) == ["control (upper bound)", "control (upper bound)", "control (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
    "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality"]

# Infeasible controls
solver_inf = Solver(model, obj_con, N=N)
solver_inf.opts.infeasible = true
@test solver_inf.opts.constrained == true
@test get_num_constraints(solver_inf) == (p+n,pI,pE+n)
@test original_constraint_inds(solver_inf) == [trues(p); falses(n)]
@test get_constraint_labels(solver_inf) == ["control (upper bound)", "control (upper bound)", "control (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
    "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality",
    "* infeasible control","* infeasible control","* infeasible control"]

# Minimum time
obj_mintime = update_objective(obj_con, tf=:min)
solver_min = Solver(model, obj_mintime, N=N)
@test solver_min.opts.constrained == true
@test get_num_constraints(solver_min) == (p+3,pI+2,pE+1)
@test original_constraint_inds(solver_min) == [true; true; false; true; false; trues(4); trues(5); false]
@test get_constraint_labels(solver_min) == ["control (upper bound)", "control (upper bound)", "* √dt (upper bound)", "control (lower bound)", "* √dt (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
    "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality",
    "* √dt (equality)"]

# Minimum time and infeasible
obj_mintime = update_objective(obj_con, tf=:min)
solver_min = Solver(model, obj_mintime, N=N)
solver_min.opts.infeasible = true
@test solver_min.opts.constrained == true
@test get_num_constraints(solver_min) == (p+3+n,pI+2,pE+1+n)
@test original_constraint_inds(solver_min) == [true; true; false; true; false; trues(4); trues(5); falses(4)]
@test get_constraint_labels(solver_min) == ["control (upper bound)", "control (upper bound)", "* √dt (upper bound)", "control (lower bound)", "* √dt (lower bound)", "state (upper bound)", "state (upper bound)", "state (upper bound)", "state (lower bound)",
    "custom inequality", "custom inequality", "custom equality", "custom equality", "custom equality",
    "* infeasible control","* infeasible control","* infeasible control","* √dt (equality)"]
get_constraint_labels(solver_min)
