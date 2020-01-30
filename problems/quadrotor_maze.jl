# Quadrotor in Maze

function QuadrotorMaze(::Type{Rot}; use_rot=true, costfun=:Quadratic) where Rot<:Rotation

# model
model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
n,m = size(model)

N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

x0 = Dynamics.build_state(model, [0,0,10], I(UnitQuaternion), zeros(3), zeros(3))
xf = Dynamics.build_state(model, [0,60,10], I(UnitQuaternion), zeros(3), zeros(3))

# cost
costfun == :QuatLQR ? sq = 0 : sq = 1
Q_diag = Dynamics.fill_state(model, 1e-3, 1e-2*sq, 1e-3, 1e-3)
R_diag = @SVector fill(1e-4,m)
Q = Diagonal(Q_diag)
R = Diagonal(R_diag)
Qf = Diagonal(@SVector fill(1e3,n))

if costfun == :Quadratic
    obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs
else
    cost = QuatLQRCost(Q, R, xf, w=1e-3)
    obj = Objective(cost, N)
end

# constraints
r_quad_maze = 2.0
r_cylinder_maze = 2.0
maze_cylinders = []
zh = 3
l1 = 5
l2 = 4
l3 = 4
l4 = 10

for i = range(-25,stop=-10,length=l1)
    push!(maze_cylinders,(i, 10,r_cylinder_maze))
end

for i = range(10,stop=25,length=l1)
    push!(maze_cylinders,(i, 10, r_cylinder_maze))
end

for i = range(-5,stop=5,length=l3)
    push!(maze_cylinders,(i, 30, r_cylinder_maze))
end

for i = range(-25,stop=-10,length=l1)
    push!(maze_cylinders,(i, 50, r_cylinder_maze))
end

for i = range(10,stop=25,length=l1)
    push!(maze_cylinders,(i, 50, r_cylinder_maze))
end

for i = range(10+2*r_cylinder_maze,stop=50-2*r_cylinder_maze,length=l4)
    push!(maze_cylinders,(-25, i, r_cylinder_maze))
end

for i = range(10+2*r_cylinder_maze,stop=50-2*r_cylinder_maze,length=l4)
    push!(maze_cylinders,(25, i, r_cylinder_maze))
end

n_maze_cylinders = length(maze_cylinders)
maze_xyr = collect(zip(maze_cylinders...))
cx = SVector{44}(maze_xyr[1])
cy = SVector{44}(maze_xyr[2])
cr = SVector{44}(maze_xyr[3])

obs = CircleConstraint(n, cx, cy, cr)

u_min = 0.
u_max = 50.
x_max = Inf*ones(n)
x_min = -Inf*ones(n)


x_max[1:3] = [25.0; Inf; 20]
x_min[1:3] = [-25.0; -Inf; 0.]

if n == 13
    noquat = @SVector [1,2,3,8,9,10,11,12,13]
    x_max[8:10] .= 30.
else
    noquat = @SVector [1,2,3,7,8,9,10,11,12]
    x_max[7:9] .= 30.
end

bnd1 = BoundConstraint(n,m,u_min=u_min,u_max=u_max)
bnd2 = BoundConstraint(n,m,u_min=u_min,u_max=u_max,x_min=x_min,x_max=x_max)
goal = GoalConstraint(xf, noquat)


U_hover = [0.5*9.81/4.0*ones(m) for k = 1:N-1] # initial hovering control trajectory

conSet = ConstraintSet(n,m,N)
add_constraint!(conSet, obs, 1:N-1)
add_constraint!(conSet, bnd2, 2:N-1)
add_constraint!(conSet, goal, N:N)

quadrotor_maze = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(quadrotor_maze,U_hover); # initialize problem with controls

X_guess = zeros(n,7)
X_guess[:,1] = x0
X_guess[:,7] = xf
X_guess[1:3,2:6] .= [0   -12.5 -15 -12.5  0 ;
                     15   20    30  40   45 ;
                     10   10    10  10   10]

if n == 13
    X_guess[4:7,:] .= q0
end
X0 = interp_rows(N,tf,X_guess);
initial_states!(quadrotor_maze, X0)

quadrotor_maze_objects = (cx,cy,cr)

return quadrotor_maze
end
