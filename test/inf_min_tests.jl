model, obj = Dynamics.quadrotor
N = 21
solver = Solver(model,obj,N=N)
n,m = get_sizes(solver)
U0 = 6*ones(m,N-1)
X0 = rollout(solver,U0)

costfun = obj.cost
model_d = Model{Discrete}(model,rk4)
U = to_dvecs(U0)
X = empty_state(n,N)
x0 = obj.x0
dt = solver.dt
C = AbstractConstraint[]
prob = Problem(model_d,costfun,x0,U,dt)
X
bp = BackwardPassNew(prob)

opts = iLQRSolverOptions(iterations=50, gradient_norm_tolerance=1e-4, verbose=false)
ilqr = iLQRSolver(prob,opts)

bp.Qx[1] = obj.cost.q
bp.Qx

bp.Qx[1]
obj.cost.q
X[1]
cost_expansion!(bp,obj.cost, rand(n), rand(m), 2)
cost_expansion!(ilqr,obj.cost,rand(n))

ilqr.s[end]


e = Expansion(prob)
reset!(e)

et = [e for i = 1:5]

et2 = copy(et)

ilqr2 = copy(ilqr)
ctg2 = copy(ilqr.ctg)

copy(opts)

cost_expansion!(et, obj.cost, rand(n), rand(m), 1)
et
cost_expansion!(ilqr,obj.cost,rand(n))
