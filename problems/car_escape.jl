# Car Escape
T = Float64;

# model
model = Dynamics.car
model_d = rk3(model)
n = model.n; m = model.m
x0 = [2.5;2.5;0.]
xf = [7.5;2.5;0.]

# cost
Q = (1e-3)*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
Qf = 100.0*Diagonal(I,n)

# constraints
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
        c[i] = circle_constraint(x,circles_escape[i][1],circles_escape[i][2],circles_escape[i][3])
    end
end

trap = Constraint{Inequality}(cI_escape,n,m,n_circles_escape,:trap)
bnd = BoundConstraint(n,m,u_min=-5.,u_max=5.)
goal = goal_constraint(xf)

N = 101
tf = 3.0
U = [ones(m) for k = 1:N-1]
obj = LQRObjective(Q,R,Qf,xf,N)

constraints = Constraints(N)
constraints[1] += bnd
for k = 2:N-1
    constraints[k] += trap + bnd
end
constraints[N] += goal

car_escape = Problem(model_d,obj,constraints=constraints,N=N,tf=tf,x0=x0,xf=xf)
initial_controls!(car_escape, U);

X_guess = [2.5 2.5 0.;4. 5. .785;5. 6.25 0.;7.5 6.25 -.261;9 5. -1.57;7.5 2.5 0.]
X0_escape = interp_rows(N,tf,Array(X_guess'))

copyto!(car_escape.X,X0_escape)

# plot escape
function plot_escape(X,x0=x0,xf=xf,X0=X0_escape)
    X_array = to_array(X)
    plot(labels="")
    plot_obstacles(circles_escape,:grey)
    plot!(X0[1,:],X0[2,:],width=2,color=:purple,label="X_guess",axis=:off)
    plot!((x0[1],x0[2]),marker=:circle,color=:red,label="")
    plot!((xf[1],xf[2]),marker=:circle,color=:green,label="",legend=:left)
    plot!(X_array[1,:],X_array[2,:],color=:blue,width=2,label="Solution")
end


# Static Car Escape
T = Float64;

# model
model = Dynamics.DubinsCar()
n,m = size(model)
x0 = @SVector [2.5,2.5,0.]
xf = @SVector [7.5,2.5,0.]
N = 101
tf = 3.0

# cost
Q = (1e-3)*Diagonal(@SVector ones(n))
R = (1e-2)*Diagonal(@SVector ones(m))
Qf = 100.0*Diagonal(@SVector ones(n))
obj = LQRObjective(Q,R,Qf,xf,N)

# constraints
r = 0.5
s1 = 30; s2 = 50; s3 = 15

circles_escape = NTuple{3,Float64}[]

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

circles_escape
x,y,r = collect(zip(circles_escape...))
x = SVector{n_circles_escape}(x)
y = SVector{n_circles_escape}(y)
r = SVector{n_circles_escape}(r)

obs = CircleConstraint(n,m,x,y,r)
con_obs = ConstraintVals(obs, 2:N-1)

bnd = StaticBoundConstraint(n,m,u_min=-5.,u_max=5.)
con_bnd = ConstraintVals(bnd, 1:N-1)

goal = GoalConstraint(n,m,xf)
con_xf = ConstraintVals(goal, N:N)

conSet = ConstraintSets(n,m,[con_obs, con_bnd, con_xf], N)

# Build problem
U0 = [@SVector ones(m) for k = 1:N-1]

car_escape_static = StaticProblem(model, obj, xf, tf;
    constraints=conSet, x0=x0)
initial_controls!(car_escape_static, U0);

X_guess = [2.5 2.5 0.;
           4. 5. .785;
           5. 6.25 0.;
           7.5 6.25 -.261;
           9 5. -1.57;
           7.5 2.5 0.]
X0_escape = interp_rows(N,tf,Array(X_guess'))
initial_states!(car_escape_static, X0_escape)
