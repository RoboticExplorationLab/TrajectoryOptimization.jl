# Quadrotor in Maze
T = Float64

# model
model = Dynamics.quadrotor_model
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
Q = (1.0e-4)*Diagonal(I,n)
R = (1.0e-3)*Diagonal(I,m)
Qf = 1000.0*Diagonal(I,n)

# constraints
r_quad_wall = 2.
r_cylinder_wall = 2.
wall_cylinders = []
zh = 3
l1 = 5
l2 = 4
l3 = 3
l4 = 10

# for i = range(-25,stop=-10,length=l1)
#     push!(wall_cylinders,(i, 10,r_cylinder_wall))
# end
#
# for i = range(10,stop=25,length=l1)
#     push!(wall_cylinders,(i, 10, r_cylinder_wall))
# end

for i = range(-6,stop=6,length=l3)
    push!(wall_cylinders,(i, 30, r_cylinder_wall))
end

# for i = range(-25,stop=-10,length=l1)
#     push!(wall_cylinders,(i, 50, r_cylinder_wall))
# end
#
# for i = range(10,stop=25,length=l1)
#     push!(wall_cylinders,(i, 50, r_cylinder_wall))
# end
#
# for i = range(10+2*r_cylinder_wall,stop=50-2*r_cylinder_wall,length=l4)
#     push!(wall_cylinders,(-25, i, r_cylinder_wall))
# end
#
# for i = range(10+2*r_cylinder_wall,stop=50-2*r_cylinder_wall,length=l4)
#     push!(wall_cylinders,(25, i, r_cylinder_wall))
# end

n_wall_cylinders = length(wall_cylinders)

function cI_wall(c,x,u)
    for i = 1:n_wall_cylinders
        c[i] = circle_constraint(x,wall_cylinders[i][1],wall_cylinders[i][2],wall_cylinders[i][3]+r_quad_wall)
    end
end

wall = Constraint{Inequality}(cI_wall,n,m,n_wall_cylinders,:wall)

u_min = 0.
u_max = 10.
x_max = Inf*ones(model.n)
x_min = -Inf*ones(model.n)

x_max[3] = 20.
x_min[3] = 0.
bnd = BoundConstraint(n,m,u_min=u_min,u_max=u_max,x_min=x_min,x_max=x_max,trim=true)

goal = goal_constraint(xf)



N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

U_hover = [0.5*9.81/4.0*ones(m) for k = 1:N-1] # initial hovering control trajectory
obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs
constraints = Constraints([bnd,wall],N) # constraint trajectory

quadrotor_wall_problem = Problem(model_d, obj, constraints=constraints, x0=x0, xf=xf, N=N, dt=dt)
quadrotor_wall_problem.constraints[N] += goal
initial_controls!(quadrotor_wall_problem,U_hover); # initialize problem with controls

X_guess = zeros(n,7)
X_guess[:,1] = x0
X_guess[:,7] = xf
X_guess[1:3,2:6] .= [0 -12.5 -20 -12.5 0 ;15 20 30 40 45 ;10 10 10 10 10]

X_guess[4:7,:] .= q0
X0 = interp_rows(N,tf,X_guess);
copyto!(quadrotor_wall_problem.X,X0)
