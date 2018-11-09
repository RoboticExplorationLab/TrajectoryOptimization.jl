using Test
### foh augmented dynamics
# Set up Dubins car system
dt = 0.1
n_dc = 3
m_dc = 2
model_dc = TrajectoryOptimization.Dynamics.dubinscar![1]
obj_uncon_dc = TrajectoryOptimization.Dynamics.dubinscar![2]

# Use continuous dynamics to create augmented continuous dynamics function
fc! = model_dc.f
fc_aug! = TrajectoryOptimization.f_augmented!(fc!,m_dc,n_dc)
fd! = TrajectoryOptimization.rk3_foh(fc!,dt)
fd_aug! = TrajectoryOptimization.fd_augmented_foh!(fd!,n_dc,m_dc)

x = ones(n_dc)
u1 = ones(m_dc)
u2 = ones(m_dc)

# test that normal dynamics and augmented dynamics outputs match
@test norm(fd!(zeros(n_dc),x,u1,u2,dt) - fd_aug!(zeros(n_dc+m_dc+1+m_dc+1),[x;u1;sqrt(dt);u2;sqrt(dt)])[1:n_dc,1]) < 1e-5
###

### Continuous dynamics Jacobians match known analytical solutions
solver_test = TrajectoryOptimization.Solver(model_dc, obj_uncon_dc, dt=dt,integration=:rk3_foh)

x = [0.0; 0.0; pi/4.0]
u = [2.0; 2.0]

Ac_known = [0.0 0.0 -sin(x[3])*u[1]; 0.0 0.0 cos(x[3])*u[1]; 0.0 0.0 0.0]
Bc_known = [cos(x[3]) 0.0; sin(x[3]) 0.0; 0.0 1.0]

Ac, Bc = solver_test.Fc(x,u)

# Compared continous dynamics Jacobian from ForwardDiff to known analytical Jacobian
@test norm((Ac_known - Ac)[:]) < 1e-5
@test norm((Bc_known - Bc)[:]) < 1e-5
###

### General constraint Jacobians match known solutions
pI = 6
n = 3
m = 3

function cI(cdot,x,u)
    cdot[1] = x[1]^3 + u[1]^2
    cdot[2] = x[2]*u[2]
    cdot[3] = x[3]
    cdot[4] = u[1]^2
    cdot[5] = u[2]^3
    cdot[6] = u[3]
end

c_jac = TrajectoryOptimization.generate_general_constraint_jacobian(cI,pI,0,n,m)

x = [1;2;3]
u = [4;5;6]

A = zeros(6,3)
B = zeros(6,3)

# cx, cu = c_jac(x,u)
c_jac(A,B,x,u)
cx_known = [3 0 0; 0 5 0; 0 0 1; 0 0 0; 0 0 0; 0 0 0]
cu_known = [8 0 0; 0 2 0; 0 0 0; 8 0 0; 0 75 0; 0 0 1]

@test all(A .== cx_known)
@test all(B .== cu_known)
###

### Custom equality constraint on quadrotor quaternion state: sqrt(q1^2 + q2^2 + q3^2 + q4^2) == 1
opts = TrajectoryOptimization.SolverOptions()
opts.verbose = false
opts.constraint_tolerance = 1e-3
opts.cost_intermediate_tolerance = 1e-3
opts.cost_tolerance = 1e-3
opts.τ = .75
opts.γ = 2
solver = TrajectoryOptimization.Solver(model,obj_con,integration=:rk4,dt=dt,opts=opts)
######################

### Set up model, objective, solver ###
# Model
n = 13 # states (quadrotor w/ quaternions)
m = 4 # controls
dt = 0.05
model, obj_uncon = TrajectoryOptimization.Dynamics.quadrotor

# -control limits
u_min = -50.0
u_max = 50.0

# -constraint that quaternion should be unit
function cE(cdot,x,u)
    cdot[1] = sqrt(x[4]^2 + x[5]^2 + x[6]^2 + x[7]^2) - 1.0
end

obj_con = TrajectoryOptimization.ConstrainedObjective(obj_uncon, u_min=u_min, u_max=u_max, cE=cE)#,cI=cI)

# Solver
# - Initial control and state trajectories
U = 10.0*ones(solver.model.m, solver.N)
##################

### Solve ###
results, stats = TrajectoryOptimization.solve(solver,U)
#############
@test stats["c_max"][end] < opts.constraint_tolerance
