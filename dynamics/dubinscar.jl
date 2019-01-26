## Dubins car
# TODO test and inplace dynamics

function dubins_dynamics!(xdot,x,u)
    xdot[1] = u[1]*cos(x[3])
    xdot[2] = u[1]*sin(x[3])
    xdot[3] = u[2]
    xdot
end
n,m = 3,2

model = Model(dubins_dynamics!,n,m)

##########################
## Constrained Examples ##
##########################

# Unconstrained Parallel Park
# initial and goal states
x0 = [0.;0.;0.]
xf = [0.;1.;0.]

# costs
Q = (1e-2)*Diagonal(I,n)
Qf = 100.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

# simulation
tf = 5.0
dt = 0.01

obj_uncon = LQRObjective(Q, R, Qf, tf, x0, xf)

dubinscar = [model, obj_uncon]

## Box Parallel Park
x0 = [0.0;0.0;0.]
xf = [0.0;1.0;0.]
tf =  3.
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

x_min = [-0.25; -0.001; -Inf]
x_max = [0.25; 1.001; Inf]

obj_con_box = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max)

dubinscar_parallelpark = [model, obj_con_box]

## Obstacle Avoid (x3)
x0 = [0.0;0.0;0.]
xf = [10.0;10.0;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

u_min = [0.; -100.]
u_max = [5.75; 100.]
x_min = [-Inf; -Inf; -Inf]
x_max = [Inf; Inf; Inf]

n_circles_3obs = 3
r_3obs = 1.0
circles_3obs = ((2.5,2.5,r_3obs),(5.,5.,r_3obs),(7.5,7.5,r_3obs))

function cI_3obs(c,x,u)
    for i = 1:n_circles_3obs
        c[i] = TrajectoryOptimization.circle_constraint(x,circles_3obs[i][1],circles_3obs[i][2],circles_3obs[i][3])
    end
    c
end

obj_con_obstacles = TrajectoryOptimization.ConstrainedObjective(obj,x_min=x_min,x_max=x_max,cI=cI_3obs)
obj_con_obstacles_control = TrajectoryOptimization.ConstrainedObjective(obj,u_min=u_min,u_max=u_max,x_min=x_min,x_max=x_max,cI=cI_3obs)

dubinscar_obstacles = [model, obj_con_obstacles,circles_3obs]
dubinscar_obstacles_control_limits = [model, obj_con_obstacles_control,circles_3obs]

## Escape
x0 = [2.5;2.5;0.]
xf = [7.5;2.5;0.]
tf =  3.0
Qf = 100.0*Diagonal(I,n)
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
u_min = [-5.; -5.]
u_max = [5.; 5.]

obj = LQRObjective(Q, R, Qf, tf, x0, xf)

r = 0.5
s1 = 30; s2 = 50; s3 = 15
circles_escape = []
for i in range(0,stop=5,length=s1)
    push!(circles_escape,(0.,i,r))
end
for i in range(0,stop=5,length=s1)
    push!(circles_escape,(5.,i,r))
end
for i in range(0,stop=5,length=s1)
    push!(circles_escape,(10.,i,r))
end
for i in range(0,stop=10,length=s2)
    push!(circles_escape,(i,0.,r))
end
for i in range(0,stop=3,length=s3)
    push!(circles_escape,(i,5.,r))
end
for i in range(5,stop=8,length=s3)
    push!(circles_escape,(i,5.,r))
end

n_circles_escape = 3*s1 + s2 + 2*s3

function cI_escape(c,x,u)
    for i = 1:n_circles_escape
        c[i] = TrajectoryOptimization.circle_constraint(x,circles_escape[i][1],circles_escape[i][2],circles_escape[i][3])
    end
    c
end

obj_escape = TrajectoryOptimization.ConstrainedObjective(obj,cI=cI_escape,u_min=u_min,u_max=u_max)

dubinscar_escape = [model,obj_escape,circles_escape]
