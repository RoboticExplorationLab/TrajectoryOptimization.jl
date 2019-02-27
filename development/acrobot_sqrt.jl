function acrobot_dynamics!(Ẋ, X, U)
    m1 = 1.0; #mass of link 1
    m2 = 1.0; #mass of link 2
    L1 = 1.0; #length of link 1
    L2 = 1.0; #length of link 2
    Lc1 = 0.5; #distance from pivot to center of link 1
    Lc2 = 0.5; #distance from pivot to center of link 2
    J1 = (1.0/3.0)*m1*L1*L1; #moment of inertia of link 1 about pivot
    J2 = (1.0/3.0)*m2*L2*L2; #moment of inertia of link 2 about pivot
    g = 9.81;

    q = X[1:2];
    qd = X[3:4];

    if isfinite(q[2])
        s = sin(q[2])
        c = cos(q[2])
    else
        s = Inf
        c = Inf
    end

    s1 = sin(q[1]);
    s12 = sin(q[1]+q[2]);
    s2 = sin(q[2]);
    c2 = cos(q[2]);

    H = [J1+J2+m2*L1*L1+2*m2*L1*Lc2*c2  J2+m2*L1*Lc2*c2;  J2+m2*L1*Lc2*c2  J2];
    C = [-2*m2*L1*Lc2*s2*qd[2]  -m2*L1*Lc2*s2*qd[2];  m2*L1*Lc2*s2*qd[1]  0];
    G = [(m1*Lc1 + m2*L1)*g*s1+m2*g*L2*s12;  m2*g*L2*s12];
    B = [0; 1];

    qdd = -H\(C*qd + G - B*U');

    Ẋ[1:2] = qd
    Ẋ[3:4] = qdd
    return nothing
end

n = 4
m = 1
model = Model(acrobot_dynamics!,n,m)

x0 = [0.0;0.0;0.0;0.0]
xf = [pi;0.0;0.0;0.0]

Q = 1.0*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 0.1*Diagonal(I,m)

dt = 0.05
tf = 10.0
N = tf/dt+1

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
opts = SolverOptions()
opts.verbose = true;
opts.square_root = true;

solver = Solver(model,obj_uncon,dt=dt,integration=:rk3,opts=opts);
U0 = rand(solver.model.m,solver.N-1);
@time results, stats = solve(solver,U0);

plot(results.X,label="")
plot(results.U)

u_min = -5
u_max = 5
obj_con = ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,use_xf_equality_constraint=false)


solver = Solver(model,obj_con,dt=dt,integration=:rk3,opts=opts);
U0 = rand(solver.model.m,solver.N-1);
@time results, stats = solve(solver,U0);

plot(results.X,label="")
plot(results.U)

## dircol
dircol_options = Dict("tol"=>solver.opts.cost_tolerance,"constr_viol_tol"=>solver.opts.constraint_tolerance)
X0 = rollout(solver,U0)
@time results_dircol, stats_dircol = TrajectoryOptimization.solve_dircol(solver, X0, U0, options=dircol_options)

plot(results_dircol.X',label="")
plot(results_dircol.U[:,1:solver.N-1]')

## Square root
x0 = [0.0;0.0;0.0;0.0]
xf = [pi;0.0;0.0;0.0]

Q = 1.0*Diagonal(I,n)
Q[1,1] = 1e4
Q[2,2] = 1e-12
Q[3,3] = 1e4
Q[4,4] = 1e-12
Qf = 1000.0*Diagonal(I,n)
Qf[1,1] = 1e6
Qf[2,2] = 1e-6
Qf[3,3] = 1.0
Qf[4,4] = 1e-6
R = 1e-6*Diagonal(I,m)

dt = 0.05
tf = 10.0
N = tf/dt+1

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)
opts = SolverOptions()
opts.verbose = true;
opts.square_root = false;

solver = Solver(model,obj_uncon,dt=dt,integration=:rk3,opts=opts);
U0 = rand(solver.model.m,solver.N-1);
@time results, stats = solve(solver,U0);

plot(results.X,label="")
plot(results.U)

norm(results.X[end] - solver.obj.xf)

u_min = -5
u_max = 5
obj_con = ConstrainedObjective(obj_uncon,u_min=u_min,u_max=u_max,use_xf_equality_constraint=false)

opts.constraint_tolerance = 1e-12
opts.penalty_max = 1e12
opts.penalty_scaling = 100.0
solver = Solver(model,obj_con,dt=dt,integration=:rk3,opts=opts);
solver.opts.square_root = false
solver_sqrt = Solver(model,obj_con,dt=dt,integration=:rk3,opts=opts);
solver_sqrt.opts.square_root = true
U0 = rand(solver.model.m,solver.N-1);
@time results, stats = solve(solver,U0);
@time results_sqrt, stats_sqrt = solve(solver_sqrt,U0);

plot(stats["c_max"])
plot!(stats_sqrt["c_max"])

plot(stats["S condition number"])
plot!(stats_sqrt["S condition number"])
