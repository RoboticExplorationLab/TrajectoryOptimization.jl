# #  Car w/ obstacles
# T = Float64
# model = Dynamics.car
# model_d = rk3(model)
# n = model.n # number of states
# m = model.m; # number of controls
#
# x0 = [0.; 0.; 0.]
# xf = [1.; 1.; 0.]
#
# Q = (1.0)*Diagonal(I,n)
# R = (1.0e-1)*Diagonal(I,m)
# Qf = 100.0*Diagonal(I,n)
#
# # create obstacle constraints
# r_circle_3obs = 0.1
# circles_3obs = ((0.25,0.25,r_circle_3obs),(0.5,0.5,r_circle_3obs),(0.75,0.75,r_circle_3obs))
# n_circles_3obs = length(circles_3obs)
#
# function circle_obs(c,x,u)
#     for i = 1:n_circles_3obs
#         c[i] = TrajectoryOptimization.circle_constraint(x,circles_3obs[i][1],circles_3obs[i][2],circles_3obs[i][3])
#     end
#     return nothing
# end
#
# obs = Constraint{Inequality}(circle_obs,n,m,n_circles_3obs,:obs);
# goal = goal_constraint(xf)
#
# N = 101 # number of knot points
# dt = 0.05 # total time
#
# U = [0.01*ones(m) for k = 1:N-1]
# obj = LQRObjective(Q,R,Qf,xf,N)
#
# obj = LQRObjective(Q,R,Qf,xf,N) # objective with same stagewise costs
#
# goal = goal_constraint(xf)
# constraints = Constraints(N) # constraints at each stage
# for k = 2:N-1
#     constraints[k] += obs
# end
# constraints[N] += goal
#
# car_3obs = Problem(model_d,obj, constraints=constraints, x0=x0, N=N, dt=dt, xf=xf)
#
# initial_controls!(car_3obs,U); # initialize problem with controls
#
# function plot_car_3obj(X,x0=x0,xf=xf; kwargs...)
#     X_array = to_array(X)
#
#     plot()
#     plot_obstacles(circles_3obs,:orange)
#     plot!((x0[1],x0[2]),marker=:circle,color=:red,labels="")
#     plot!((xf[1],xf[2]),marker=:circle,color=:green,labels="")
#     plot!(X_array[1,:],X_array[2,:],color=:blue,width=2,label=""; kwargs...)
# end

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

bnd = StaticBoundConstraint(n,m, u_min=[-1,-3],u_max=[2,3])
con_bnd = ConstraintVals(bnd, 1:N-1)

goal_con = GoalConstraint(n,m,xf)
con_xf = ConstraintVals(goal_con, N:N)

conSet = ConstraintSets(n,m,[con_obs, con_xf], N)

# Create problem
U = [@SVector fill(0.01,m) for k = 1:N-1]
car_3obs_static = StaticProblem(model, obj, xf, tf, constraints=conSet, x0=x0)
initial_controls!(car_3obs_static, U)
