# Quadrotor in Maze

function QuadrotorMaze(::Type{Rot}; use_rot=true, costfun=:Quadratic,
        normcon=false) where Rot<:Rotation

# model
model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
n,m = size(model)

N = 101 # number of knot points
tf = 5.0
dt = tf/(N-1) # total time

x0 = Dynamics.build_state(model, [0,-15,1], I(UnitQuaternion), zeros(3), zeros(3))
xf = Dynamics.build_state(model, [0, 15,1], I(UnitQuaternion), zeros(3), zeros(3))
utrim = Dynamics.trim_controls(model)

# cost
if n == 13
    rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
else
    rm_quat = @SVector [1,2,3,4,5,6,7,8,9,10,11,12]
end
costfun == :QuatLQR ? sq = 0 : sq = 1
Q_diag = Dynamics.fill_state(model, 1e-3, 1e-2*sq, 1e-3, 1e-3)
R_diag = @SVector fill(1e-4,m)
Q = Diagonal(Q_diag)
R = Diagonal(R_diag)
Qf = Diagonal(@SVector fill(1e3,n))

if costfun == :Quadratic
    cost = LQRCost(Q, R, xf)
    obj = Objective(cost, N)
    # obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs
elseif costfun == :QuatLQR
    cost = QuatLQRCost(Q, R, xf, w=1e-1)
    cost_term = QuatLQRCost(Qf, R, xf, w=1.0)
    obj = Objective(cost, cost_term, N)
elseif costfun == :ErrorQuad
    cost = ErrorQuadratic(model, Diagonal(Q_diag[rm_quat]), R, xf)
    cost_term = ErrorQuadratic(model, Diagonal(diag(Qf)[rm_quat]), R, xf)
    obj = Objective(cost, cost_term, N)
end

# constraints
r_quad_maze = 1.0
r_cylinder_maze = 1.2
maze_cylinders = []
zh = 3
l1 = 5
l2 = 4
l3 = 7
l4 = 10

d = 3  # 0.5 door width
w = 10  # y location of wall
mid = 3  # 0.5 middle width

x_enter=-10
x_mid=0
x_exit=10

for i = range(-w,stop=-d,length=l1) # enter wall
    push!(maze_cylinders,(i, x_enter,r_cylinder_maze))
end

for i = range(d,stop=w,length=l1) # enter wall
    push!(maze_cylinders,(i, x_enter, r_cylinder_maze))
end

for i = range(-mid,stop=mid,length=l3) # middle obstacle
    push!(maze_cylinders,(i, x_mid, r_cylinder_maze))
end

for i = range(-w,stop=-d,length=l1) # exit wall
    push!(maze_cylinders,(i, x_exit, r_cylinder_maze))
end

for i = range(d,stop=w,length=l1) # exit wall
    push!(maze_cylinders,(i, x_exit, r_cylinder_maze))
end

# Top and bottom walls
for i = range(x_enter+2*r_cylinder_maze,stop=x_exit-2*r_cylinder_maze,length=l4)
    push!(maze_cylinders,(-w, i, r_cylinder_maze))
end

for i = range(x_enter+2*r_cylinder_maze,stop=x_exit-2*r_cylinder_maze,length=l4)
    push!(maze_cylinders,(w, i, r_cylinder_maze))
end

n_maze_cylinders = length(maze_cylinders)
maze_xyr = collect(zip(maze_cylinders...))

cx = SVector{n_maze_cylinders}(maze_xyr[1])
cy = SVector{n_maze_cylinders}(maze_xyr[2])
cr = SVector{n_maze_cylinders}(maze_xyr[3])

obs = CircleConstraint(n, cx, cy, cr .+ r_quad_maze)

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

u0 = @SVector fill(0.5*9.81/4.0, m)
U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

conSet = ConstraintSet(n,m,N)
add_constraint!(conSet, obs, 1:N-1)
add_constraint!(conSet, bnd2, 2:N-1)
add_constraint!(conSet, goal, N:N)

# Quaternion norm constraint
if normcon
    if use_rot == :slack
        add_constraint!(conSet, QuatSlackConstraint(), 1:N-1)
    else
        add_constraint!(conSet, QuatNormConstraint(), 1:N-1)
        u0 = [u0; (@SVector [1.])]
    end
end

quadrotor_maze = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(quadrotor_maze,U_hover); # initialize problem with controls

X_guess = zeros(n,7)
X_guess[:,1] = x0
X_guess[:,7] = xf
wpts = [2 -7 1;
        5 -5 1;
        7  0 1;
        5  5 1;
        2  7 1;]
X_guess[1:3,2:6] .= wpts'

if n == 13
    X_guess[4:7,:] .= q0
end
X0 = interp_rows(N,tf,X_guess);
initial_states!(quadrotor_maze, X0)

quadrotor_maze_objects = (cx,cy,cr)

return quadrotor_maze
end
