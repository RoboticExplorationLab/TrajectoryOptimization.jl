model = Dynamics.pendulum[1]

# Objective
n,m = model.n, model.m
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)
Q = 1e-3*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = 1e-2*Diagonal(I,m)
tf = 5.
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

N = 41
U0 = ones(m,N)

# Test original problem
solver = Solver(model, obj_uncon, N=N)
res, stats = solve(solver, U0)
@test norm(res.X[N]-xf) < 1e-3

# Convert LQRObjective to Generic Objective
ell(x,u) = 0.5*x'Q*x + 0.5*u'R*u - xf'Q*x
ell(x) = 0.5*x'Qf*x - xf'Qf*x

costfun = GenericCost(ell,ell,n,m)
obj_generic = UnconstrainedObjectiveNew(costfun,tf,x0,xf)
solver_g = Solver(model, obj_generic, N=N)
res_g, stats_g = solve(solver_g, U0)
@test norm(res_g.X - res.X) == 0

# Try a non-linear cost function
ell2(x,u) = cos(x[1]) + u'R*u + x[2]^2*1e-3
ell2(x) = cos(x[1]) + x[2]^2*1e-1

costfun2 = GenericCost(ell2,ell2,n,m)
obj2 = UnconstrainedObjectiveNew(costfun2,tf,x0,xf)
solver_g2 = Solver(model, obj2, N=N)
res_g2, stats_g2 = solve(solver_g2, U0*0)
@test norm(res_g2.X[N] - xf) < 1e-6

# Constrained (non-zero initialization)
x_min = [-2*pi,-Inf]
x_max = [2*pi,Inf]
obj2_con = ConstrainedObjectiveNew(obj2,x_min=x_min,x_max=x_max)
solver_g2 = Solver(model, obj2_con, N=N)
res_g2_con, stats_g2_con = solve(solver_g2, U0)
@test norm(res_g2_con.X[N] - xf) < 1e-3
