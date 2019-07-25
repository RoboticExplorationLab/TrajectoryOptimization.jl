# Quadrotor in Maze
T = Float64

# model
model = Dynamics.quadrotor
# model = Dynamics.quadrotor_euler
model_d = rk3(model)
n = model.n; m = model.m
q0 = [1.;0.;0.;0.] # unit quaternion

x0 = zeros(T,n)
x0[1:3] = [0.; 0.; 10.]
x0[4:7] = q0

xf = zero(x0)
xf[1:3] = [0.;60.; 10.]
xf[4:7] = q0;

# cost
Q = (1.0e-3)*Diagonal(I,n)
Q[4:7,4:7] = (1.0e-2)*Diagonal(I,4)
R = (1.0e-4)*Diagonal(I,m)
Qf = 1000.0*Diagonal(I,n)

u_min = 0.
u_max = 50.
x_max = Inf*ones(model.n)
x_min = -Inf*ones(model.n)

x_max[1:3] = [25.0; Inf; 20]
x_min[1:3] = [-25.0; -Inf; 0.]
bnd3 = BoundConstraint(n,m,u_min=u_min)

xf_no_quat_U = copy(xf)
xf_no_quat_L = copy(xf)
xf_no_quat_U[4:7] .= Inf
xf_no_quat_L[4:7] .= -Inf
xf_no_quat_U[8:10] .= 0.
xf_no_quat_L[8:10] .= 0.
bnd_xf = BoundConstraint(n,m,x_min=xf_no_quat_L,x_max=xf_no_quat_U)

N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

U_hover = [0.5*9.81/4.0*ones(m) for k = 1:N-1] # initial hovering control trajectory
obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs


quadrotor = Problem(model_d, obj, x0=x0, xf=xf, N=N, dt=dt)
initial_controls!(quadrotor,U_hover); # initialize problem with controls

quadrotor.constraints[1] += bnd3
for k = 2:N-1
    quadrotor.constraints[k] += bnd3
end
quadrotor.constraints[N] += bnd_xf
