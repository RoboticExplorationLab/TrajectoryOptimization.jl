using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using Random
using Distributions

opts_ilqr = iLQRSolverOptions(verbose=false,
    cost_tolerance=1e-4,
    iterations=300)

opts_al = AugmentedLagrangianSolverOptions(verbose=false,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=1e-4,
    penalty_scaling=10.,
    penalty_initial=1.)

# Solve Quadrotor zig-zag problem
Rot = MRP{Float64}
prob = Problems.gen_quadrotor_zigzag(Rot, costfun=:Quadratic, use_rot=false)
solver = iLQRSolver(prob)
solve!(solver)
iterations(solver)
@test norm(state(solver.Z[end])[1:3] - solver.xf[1:3]) < 0.1

# Convert trajectory to general RBState
Xref = Controllers.RBState(solver.model, solver.Z)
Uref = controls(solver)
push!(Uref, Dynamics.trim_controls(solver.model))
dt_ref = solver.Z[1].dt

# Track with different model
Rot = UnitQuaternion{Float64,CayleyMap}
model = Dynamics.Quadrotor2{Rot}(use_rot=true, mass=.5)
dt = 1e-4
Nsim = Int(solver.tf/dt) + 1
inds = Int.(range(1,Nsim,length=solver.N))
Q = Diagonal(Dynamics.fill_error_state(model, 200, 200, 50, 50))
R = Diagonal(@SVector fill(1.0, 4))
mlqr = Controllers.TVLQR(model, Q, R, Xref, Uref, dt_ref)

noise = MvNormal(Diagonal(fill(5.,6)))
model_ = Dynamics.NoisyRB(model, noise)
X = Controllers.simulate(model_, mlqr, Xref[1], solver.tf, w=0.0)
res_lqr = [Controllers.RBState(model, x) for x in X[inds]]
e1 = norm(res_lqr .⊖ Xref)


# Track with SE3 controller

# get reference trajectories
b_ = map(Xref) do x
    # prefer pointing torwards the goal
    x.r - Xref[end].r + 0.01*@SVector rand(3)
end
X_ = map(Xref) do x
    Dynamics.build_state(model, x)
end
Xd_ = map(1:solver.N) do k
    dynamics(model, X_[k], Uref[k])
end
t = collect(0:dt_ref:solver.tf)

# cntrl = SE3Tracking(model, X_, Xd_, b_, t)
kx = 2.71 + 10.0
kv = 1.01 + 3.0
kR = 3.0*Diagonal(@SVector [1,1,1.])
kO = 0.5*Diagonal(@SVector [1,1,1.])
hfca = Controllers.HFCA(model, X_, Xd_, b_, t, kx=kx, kv=kv, kR=kR, kO=kO)

X2 = Controllers.simulate(model, hfca, Xref[1], solver.tf, w=0.0)
res_so3 = [Controllers.RBState(model, x) for x in X2[inds]]
e2 = norm(res_so3 .⊖ Xref)
@test e1 < e2


# Test MLQR
model = Dynamics.Quadrotor2()
Q_ = (200.,200,50,50)
R = Diagonal(@SVector fill(1.0,4))
xref = zeros(model)[1]
uref = Dynamics.trim_controls(model)
dt_ctrl = 0.01
x0 = Controllers.RBState([0,0,0.2], expm(deg2rad(80)*@SVector [1,0,0]), [0,0,1.], [0.5,0.5,0])
tf = 10.0

Q = Diagonal(Dynamics.fill_error_state(model, Q_...))
cntrl = Controllers.MLQR(model, dt_ctrl, Q, R, xref, uref)
xinit = Dynamics.build_state(model, x0)
res = Controllers.simulate(model, cntrl, xinit, tf, dt=dt, w=0.0)
err = TrajectoryOptimization.state_diff(model, res[end], xref)
@test norm(err) < 0.01

# Test LQR
model = Dynamics.Quadrotor2{RodriguesParam{Float64}}(use_rot=false)
Q = Diagonal(Dynamics.fill_error_state(model, Q_...))
xref = zeros(model)[1]
cntrl = Controllers.LQR(model, dt_ctrl, Q, R, xref, uref)
xinit = Dynamics.build_state(model, x0)
res = Controllers.simulate(model, cntrl, xinit, tf, dt=dt, w=0.0)

# Test HFCA
Rot = UnitQuaternion{Float64,IdentityMap}
model = Dynamics.Quadrotor2{Rot}(use_rot=false)
xref = zeros(model)[1]
err = TrajectoryOptimization.state_diff(model, res[end], xref)
@test norm(err) < 0.01

times = range(0,tf,step=dt)
Xref = [copy(xref) for k = 1:length(times)]
bref = [@SVector [1,0,0.] for k = 1:length(times)]
Xdref = [@SVector zeros(13) for k = 1:length(times)]

cntrl = Controllers.HFCA(model, Xref, Xdref, bref, collect(times))
xinit = Dynamics.build_state(model, x0)
res = Controllers.simulate(model, cntrl, xinit, tf, dt=dt, w=0.0)
err = TrajectoryOptimization.state_diff(model, res[end], xref)
@test norm(err) < 0.01

# Test SE3
cntrl = Controllers.SE3Tracking(model, Xref, Xdref, bref, collect(times))
xinit = Dynamics.build_state(model, x0)
res = Controllers.simulate(model, cntrl, xinit, tf, dt=dt, w=0.0)
err = TrajectoryOptimization.state_diff(model, res[end], xref)
@test norm(err) < 0.01
