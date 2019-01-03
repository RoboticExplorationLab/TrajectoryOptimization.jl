using MeshCatMechanisms
using LinearAlgebra
using RigidBodyDynamics
using Plots
include("../kuka_visualizer.jl")
model, obj = Dynamics.kuka
n,m = model.n, model.m


function hold_trajectory(solver, mech, q)
    state = MechanismState(mech)
    set_configuration!(state, q)
    vd = zero(state.q)
    u0 = dynamics_bias(state)

    n,m,N = get_sizes(solver)
    if length(q) > m
        throw(ArgumentError("system must be fully actuated to hold an arbitrary position ($(length(q)) should be > $m)"))
    end
    U0 = zeros(m,N)
    for k = 1:N
        U0[:,k] = u0
    end
    return U0
end

# Define objective
x0 = zeros(n)
x0[2] = -pi/2
xf = copy(x0)
xf[1] = pi/4
# xf[2] = pi/2

Q = 1e-4*Diagonal(I,n)
Qf = 250.0*Diagonal(I,n)
R = 1e-4*Diagonal(I,m)

tf = 5.0
obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

# Define solver
N = 41
solver = Solver(model,obj_uncon,N=N)

# Generate initial control trajectory
kuka = parse_urdf(Dynamics.urdf_kuka)
U0 = hold_trajectory(solver, kuka, x0[1:7])
X0 = TrajectoryOptimization.rollout(solver,U0)
plot(X0[1:7,:]',ylim=[-10,10])

# Solve
solver.opts.verbose = true
solver.opts.live_plotting = false
solver.opts.iterations_innerloop = 200
solver.opts.infeasible
res, stats = solve(solver,U0)
res.X[N]-xf

visuals = URDFVisuals(Dynamics.urdf_kuka)
vis = MechanismVisualizer(kuka, visuals)
open(vis)
set_configuration!(vis, x0[1:7])
animate_trajectory(vis, res.X)
set_configuration!(vis, xf[1:7])
plot(to_array(res.U)')

state = MechanismState(kuka)
set_configuration!(state,x0[1:7])
vd = zero(state.q)
u0 = inverse_dynamics(state,vd)
dynamics_bias(state)
U0 = zeros(m,N)
for k = 1:N
    U0[:,k] = u0
end


U0 = rand(m,N)
solver.opts.verbose = true
res, stats = solve(solver,U0)

init_results(solver,Matrix{Float64}(undef,0,0),U0)
X0 = rollout(solver,U0)
plot(X0[1:6,:]',ylim=[-10,10])
X0[1:6,1]



state.q
