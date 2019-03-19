import TrajectoryOptimization: dynamics
using RigidBodyDynamics

######## Analytical Model #############
f = Dynamics.dubins_dynamics!
n,m = 3,2
model = Model(f,n,m)
@test model.m == m
@test model.n == n

xdot = zeros(n)
x = rand(n)
u = rand(m)
@test evals(model) == 0
reset(model)
model.f(xdot,x,u)
@test evals(model) == 0
dynamics(model,xdot,x,u)
@test evals(model) == 1

# Test jacobians
x,u = rand(n),rand(m)
z = [x;u]
ẋ = zeros(n)
@test_nowarn Z = model.∇f(x,u)
@test_nowarn model.∇f(Z,x,u)
@test all(TrajectoryOptimization._test_jacobian(∇f!))

# Custom dynamics
mass_ = 10
J = Diagonal(I,n)/2
n = 4
m = 2
function gen_dynamics(mass,J)
    function f2(ẋ,x,u)
        ẋ[1:2] = x[1:2] .^2 .+ u[1]
        ẋ[3:4] = mass*x[3:4] + u
        ẋ[1:4] .+= J*x
    end
    function ∇f2(Z,x,u)
        Z[1,1] = 2*x[1] + J[1,1]
        Z[1,5] = 1
        Z[2,2] = 2*x[2] + J[2,2]
        Z[2,5] = 1
        Z[3,3] = mass + J[3,3]
        Z[3,5] = 1
        Z[4,4] = mass + J[4,4]
        Z[4,6] = 1
    end
    return f2,∇f2
end
f1,∇f1 = gen_dynamics(mass_,J)

ẋ = zeros(n)
ẋ2 = zeros(n)
x,u = rand(n), rand(m)
Z = zeros(n,n+m)
∇f1(Z,x,u)
∇f1!, = generate_jacobian(f1,n,m)
@test ∇f1!(x,u) == Z
model1 = Model(f1,n,m)
model2 = Model(f1,∇f1,n,m)
@test model1.∇f(x,u) == model2.∇f(x,u)
t_fd = @elapsed model1.∇f(Z,x,u)
t_an = @elapsed model2.∇f(Z,x,u)
@test t_an*1.5 < t_fd
model1.f(ẋ,x,u)
model2.f(ẋ2,x,u)
@test ẋ == ẋ2

params = (mass=mass_,J=J);
function f3(ẋ,x,u,p)
    ẋ[1:2] = x[1:2] .^2 .+ u[1]
    ẋ[3:4] = p[:mass]*x[3:4] + u
    ẋ[1:4] .+= p[:J]*x
end
function ∇f3(Z,x,u,p)
    mass = p[:mass]
    J = p[:J]
    Z[1,1] = 2*x[1] + J[1,1]
    Z[1,5] = 1
    Z[2,2] = 2*x[2] + J[2,2]
    Z[2,5] = 1
    Z[3,3] = mass + J[3,3]
    Z[3,5] = 1
    Z[4,4] = mass + J[4,4]
    Z[4,6] = 1
end
∇f3(Z,x,u,params)
f3_p(ẋ,x,u) = f3(ẋ,x,u,params)
∇f3!, = generate_jacobian(f3_p,n,m)
@test ∇f3!(x,u) == Z
_check_dynamics(f3_p,n,m)
model3 = Model(f3,n,m,params)
model4 = Model(f3,∇f3,n,m,params)
@test model3.∇f(x,u) == model4.∇f(x,u)
@test model1.∇f(x,u) == model3.∇f(x,u)
model3.f(ẋ,x,u)
model4.f(ẋ2,x,u)
@test ẋ == ẋ2
model1.f(ẋ2,x,u)
@test ẋ == ẋ2



######### Rigid Body Dynamics Model ###############
acrobot = parse_urdf(Dynamics.urdf_doublependulum)
model = Model(acrobot)
@test evals(model) == 0
n,m = model.n, model.m
xdot = zeros(n)
x = rand(n)
u = rand(m)
dynamics(model,xdot,x,u)
@test evals(model) == 1
reset(model)
@test evals(model) == 0
