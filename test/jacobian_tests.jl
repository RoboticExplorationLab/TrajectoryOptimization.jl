using TrajectoryOptimization
using Base.Test

### foh augmented dynamics
dt = 0.1
n_dc = 3
m_dc = 2
model_dc = Dynamics.dubinscar![1]
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

c_jac = generate_general_constraint_jacobian(cI,pI,n,m)

x = [1;2;3]
u = [4;5;6]

cx, cu = c_jac(x,u)

cx_known = [3 0 0; 0 5 0; 0 0 1; 0 0 0; 0 0 0; 0 0 0]
cu_known = [8 0 0; 0 2 0; 0 0 0; 8 0 0; 0 75 0; 0 0 1]

@test all(cx .== cx_known)
@test all(cu .== cu_known)
###
