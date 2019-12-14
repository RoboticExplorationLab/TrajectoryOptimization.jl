# model
model = Dynamics.Quadrotor()
# model = Dynamics.quadrotor_euler
n,m = size(model)

# discretization
N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

q0 = @SVector [1,0,0,0]

x0_pos = @SVector [0., 0., 10.]
x0 = [x0_pos; q0; @SVector zeros(6)]

xf = zero(x0)
xf_pos = @SVector [0., 60., 10.]
xf = [xf_pos; q0; @SVector zeros(6)]

# cost
Qdiag = fill(1e-3,n)
Qdiag[4:7] .= 1e-2
Q = Diagonal(SVector{13}(Qdiag))
R = (1.0e-4)*Diagonal(@SVector ones(m))
Qf = 1000.0*Diagonal(@SVector ones(n))

u_min = 0.
u_max = 50.
x_max = Inf*ones(model.n)
x_min = -Inf*ones(model.n)

x_max[1:3] = [25.0; Inf; 20]
x_min[1:3] = [-25.0; -Inf; 0.]
bnd = StaticBoundConstraint(n,m,u_min=u_min)

xf_no_quat_U = Vector(xf)
xf_no_quat_L = Vector(xf)
xf_no_quat_U[4:7] .= Inf
xf_no_quat_L[4:7] .= -Inf
xf_no_quat_U[8:10] .= 0.
xf_no_quat_L[8:10] .= 0.
bnd_xf = StaticBoundConstraint(n,m, x_min=xf_no_quat_L, x_max=xf_no_quat_U)
inds_no_quat = SVector{n-4}(deleteat!(collect(1:n), 4:7))
goal = GoalConstraint(n,m, xf, inds_no_quat)

con_bnd = ConstraintVals(bnd, 1:N-1)
con_xf = ConstraintVals(bnd_xf, N:N)
con_goal = ConstraintVals(goal, N:N)
conSet = ConstraintSets(n,m,[con_bnd, con_goal], N)


U_hover = [0.5*9.81/4.0*(@SVector ones(m)) for k = 1:N-1] # initial hovering control trajectory
obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs

quadrotor_static = Problem(model, obj, xf, tf, constraints=conSet, U0=U_hover, x0=x0)
