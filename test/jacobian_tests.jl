using TrajectoryOptimization
using Base.Test

### foh augmented dynamics
dt = 0.1
n_dc = 3
m_dc = 2
model_dc = TrajectoryOptimization.Dynamics.dubinscar![1]
obj_uncon_dc = TrajectoryOptimization.Dynamics.dubinscar![2]

fc! = model_dc.f
fc_aug! = TrajectoryOptimization.f_augmented!(fc!,m_dc,n_dc)
fd! = TrajectoryOptimization.rk3_foh(fc!,dt)
fd_aug! = TrajectoryOptimization.f_augmented_foh!(fd!,n_dc,m_dc)

x = ones(n_dc)
u1 = ones(m_dc)
u2 = ones(m_dc)

@test norm(fd!(zeros(n_dc),x,u1,u2) - fd_aug!(zeros(n_dc+m_dc+m_dc+1),[x;u1;u2;dt])[1:n_dc,1]) < 1e-5
###

### Continuous dynamics Jacobians match known analytical solutions
solver_test = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3_foh)

x = [0.0; 0.0; pi/4.0]
u = [2.0; 2.0]

Ac_known = [0.0 0.0 -sin(x[3])*u[1]; 0.0 0.0 cos(x[3])*u[1]; 0.0 0.0 0.0]
Bc_known = [cos(x[3]) 0.0; sin(x[3]) 0.0; 0.0 1.0]

Ac, Bc = solver_test.Fc(x,u)

@test norm((Ac_known - Ac)[:]) < 1e-5
@test norm((Bc_known - Bc)[:]) < 1e-5
###

### General constraint Jacobians match known solutions
pI = 6
n = 3
m = 3

function cI(x,u)
    [x[1]^3 + u[1]^2;
     x[2]*u[2];
     x[3];
     u[1]^2;
     u[2]^3;
     u[3]]
end

c_jac = TrajectoryOptimization.generate_general_constraint_jacobian(cI,pI,n,m)

x = [1;2;3]
u = [4;5;6]

cx, cu = c_jac(x,u)

cx_known = [3 0 0; 0 5 0; 0 0 1; 0 0 0; 0 0 0; 0 0 0]
cu_known = [8 0 0; 0 2 0; 0 0 0; 8 0 0; 0 75 0; 0 0 1]

@test all(cx .== cx_known)
@test all(cu .== cu_known)
###

### Custom equality constraint on quadrotor quaternion state: sqrt(q1^2 + q2^2 + q3^2 + q4^2) == 1
n = 13 # states (quadrotor w/ quaternions)
m = 4 # controls

# Setup solver options
opts = TrajectoryOptimization.SolverOptions()
# opts.square_root = false
# opts.verbose = true
opts.cache=true
# opts.c1=1e-4
# opts.c2=3.0
# opts.mu_al_update = 10.0
# opts.eps_constraint = 1e-3
# opts.eps_intermediate = 1e-3
# opts.eps = 1e-3
# opts.outer_loop_update = :uniform
# opts.τ = 0.1
# opts.iterations_outerloop = 250
# opts.iterations = 1000

# Objective and constraints
Qf = 100.0*eye(n)
Q = 1e-1*eye(n)
R = 1e-2*eye(m)
tf = 5.0
dt = 0.1

x0 = -1*ones(n)
quat0 = TrajectoryOptimization.eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
x0[4:7] = quat0[:,1]
x0

xf = zeros(n)
xf[1:3] = [11.0;11.0;11.0] # xyz position
quatf = TrajectoryOptimization.eul2quat([0.0; 0.0; 0.0]) # ZYX Euler angles
xf[4:7] = quatf
xf

u_min = -300.0
u_max = 300.0

n_spheres = 3
spheres = ([2.5;5;7.5],[2;4;7],[2.6;2.35;7.5],[0.5;0.5;0.5])
function cI(x,u)
    [TrajectoryOptimization.sphere_constraint(x,spheres[1][1],spheres[2][1],spheres[3][1],spheres[4][1]);
     TrajectoryOptimization.sphere_constraint(x,spheres[1][2],spheres[2][2],spheres[3][2],spheres[4][2]);
     TrajectoryOptimization.sphere_constraint(x,spheres[1][3],spheres[2][3],spheres[3][3],spheres[4][3])]
end

function cE(x,u)
    [x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2 - 1.0]
end

obj_uncon = TrajectoryOptimization.TrajectoryOptimization.UnconstrainedObjective(Q, R, Qf, tf, x0, xf)
obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, cI=cI, cE=cE)

model! = TrajectoryOptimization.Model(Dynamics.quadrotor_dynamics!,n,m)

solver = TrajectoryOptimization.Solver(model!,obj_con,integration=:rk3,dt=dt,opts=opts)

U = ones(solver.model.m, solver.N)
X_interp = TrajectoryOptimization.line_trajectory(solver)

results,stats = TrajectoryOptimization.solve(solver,U)

println("Final position: $(results.X[1:3,end])\n       desired: $(obj_uncon.xf[1:3])\n    Iterations: $(stats["iterations"])\n Max violation: $(max_violation(results.result[results.termination_index]))")

# Check that quaternion state always meets constraint tolerance
@test all(results.result[results.termination_index].C .< opts.eps_constraint)
