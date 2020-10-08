
@testset "Constraint Sets" begin
model = Cartpole()
n,m = size(model)
N = 11
x,u = rand(model)
dt = 0.1
z = KnotPoint(x,u,dt)


#--- Generate some constraints
# Circle Constraint
xc = SA[1,1,1]
yc = SA[1,2,3]
r  = SA[1,1,1]
cir = CircleConstraint(n, xc, yc, r)

# Goal Constraint
xf = @SVector rand(n)
goal = GoalConstraint(xf)

# Linear Constraint
p = 5
A = @SMatrix rand(p,n+m)
b = @SVector rand(p)
lin = LinearConstraint(n,m,A,b, Inequality())

# Bound Constraint
xmin = -@SVector rand(n)
xmax = +@SVector rand(n)
umin = -@SVector rand(m)
umax = +@SVector rand(m)
bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)

# Dynamics Constraint
dyn = TO.DynamicsConstraint(model, N)

#--- Create the Constraint List
cons = ConstraintList(n,m,N)
add_constraint!(cons, cir, 1:N)
add_constraint!(cons, goal, N)
add_constraint!(cons, lin, 2:N-1)
cons2 = copy(cons)
cons2_ = deepcopy(cons)
add_constraint!(cons, bnd, 1:N-1)
add_constraint!(cons, dyn, 1:N-1)
@test length(cons) == 5
@test length(cons2) == 3
@test cons[2] === cons2[2]
@test cons[2] !== cons2_[2]


end
