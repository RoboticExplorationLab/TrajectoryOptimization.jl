
#  Car w/ obstacles
model = Dynamics.DubinsCar()
n,m = size(model)

N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1)

x0 = @SVector [0., 0., 0.]
xf = @SVector [1., 1., 0.]

Q = (1.0)*Diagonal(@SVector ones(n))
R = (1.0e-1)*Diagonal(@SVector ones(m))
Qf = 100.0*Diagonal(@SVector ones(n))
obj = LQRObjective(Q,R,Qf,xf,N)

# create obstacle constraints
r_circle_3obs = 0.1
circles_3obs = ((0.25,0.25,r_circle_3obs),(0.5,0.5,r_circle_3obs),(0.75,0.75,r_circle_3obs))
n_circles_3obs = length(circles_3obs)

circle_x = @SVector [0.25, 0.5, 0.75]
circle_y = @SVector [0.25, 0.5, 0.75]
circle_r = @SVector fill(r_circle_3obs, 3)

circle_con = CircleConstraint(n,m, circle_x, circle_y, circle_r)
con_obs = ConstraintVals(circle_con, 2:N-1)

bnd = BoundConstraint(n,m, u_min=[-1,-3],u_max=[2,3])
con_bnd = ConstraintVals(bnd, 1:N-1)

goal_con = GoalConstraint(n,m,xf)
con_xf = ConstraintVals(goal_con, N:N)

conSet = ConstraintSets(n,m,[con_obs, con_xf], N)

# Create problem
U = [@SVector fill(0.01,m) for k = 1:N-1]
car_3obs_static = Problem(model, obj, xf, tf, constraints=conSet, x0=x0)
initial_controls!(car_3obs_static, U)
