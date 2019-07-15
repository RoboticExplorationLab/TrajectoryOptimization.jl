# using MeshCatMechanisms
# using MeshCat
# using RigidBodyDynamics
# using GeometryTypes
# using CoordinateTransformations

kuka = parse_urdf(Dynamics.urdf_kuka,remove_fixed_tree_joints=false)

model = Dynamics.kuka_model
model_d = rk3(model)
n,m = model.n, model.m

x0 = zeros(n)

xf = zeros(n)
xf[1] = pi/2
xf[2] = pi/2;

Q = Diagonal([ones(7); ones(7)*100])
Qf = 1000.0*Diagonal(I,n)
R = 1e-2*Diagonal(I,m)
tf = 5.0
xf_ee = Dynamics.end_effector_function(xf)
x0_ee = Dynamics.end_effector_function(x0);

# verbose=false
# opts_ilqr = iLQRSolverOptions{T}(verbose=true,iterations=300,live_plotting=:off);
# opts_al = AugmentedLagrangianSolverOptions{T}(verbose=true,opts_uncon=opts_ilqr,
#     iterations=20,cost_tolerance=1.0e-6,cost_tolerance_intermediate=1.0e-5,constraint_tolerance=1.0e-3,penalty_scaling=50.,penalty_initial=0.01)
# opts_altro = ALTROSolverOptions{T}(verbose=true,resolve_feasible_problem=false,opts_al=opts_al,R_inf=0.01);

goal = goal_constraint(xf)

N = 41 # number of knot points
dt = tf/(N-1)# total time

U_hold = Dynamics.hold_trajectory(n,m,N, kuka, x0[1:7])
obj = LQRObjective(Q, R, Qf, xf, N)

kuka_problem = Problem(model_d, obj, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(kuka_problem, U_hold) # initialize problem with controls
kuka_problem.constraints[N] += goal
