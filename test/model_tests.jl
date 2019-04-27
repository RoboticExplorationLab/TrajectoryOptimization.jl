import TrajectoryOptimization: dynamics
import TrajectoryOptimization: Model, LQRCost, Problem, Objective, rollout!, iLQRSolverOptions,
    AbstractSolver, jacobian!, _backwardpass!, _backwardpass_sqrt!, AugmentedLagrangianSolverOptions, ALTROSolverOptions,
    bound_constraint, goal_constraint, update_constraints!, update_active_set!, jacobian!, update_problem,
    line_trajectory, total_time, generate_jacobian, _check_dynamics, AnalyticalModel, _test_jacobian,
    _check_jacobian, f_augmented!, Nominal, Uncertain

using RigidBodyDynamics
using PartedArrays

######## Analytical Model #############
model = Dynamics.car_model
n = 3; m = 2
@test model.m == m
@test model.n == n

xdot = zeros(n)
x = rand(n)
u = rand(m)
reset(model)
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
Z = BlockMatrix(model)
@test_nowarn model.∇f(Z,x,u)
@test all(TrajectoryOptimization._test_jacobian(Continuous,model.∇f))

# Custom dynamics
n = 4
m = 2
mass_ = 10
J = Diagonal(I,n)/2
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


# Custom discrete dynamics function
dt = 0.1
function gen_dynamics(mass,J,dt)
    function f2(ẋ,x,u,dt=dt)
        ẋ[1:2] = x[1:2] .^2 .+ u[1]
        ẋ[3:4] = mass*x[3:4] + u
        ẋ .+= J*x
        ẋ .= x + ẋ*dt
    end
    function ∇f2(Z,x,u::AbstractVector,dt::Float64=dt)
        Z[1,1] = (2*x[1] + J[1,1])*dt + 1
        Z[1,5] = dt
        Z[2,2] = (2*x[2] + J[2,2])*dt + 1
        Z[2,5] = dt
        Z[3,3] = (mass + J[3,3])*dt + 1
        Z[3,5] = dt
        Z[4,4] = (mass + J[4,4])*dt + 1
        Z[4,6] = dt
    end
    return f2,∇f2
end
fd1,∇fd1 = gen_dynamics(mass_,J,dt)

ẋ = zeros(n)
ẋ2 = zeros(n)
x,u = rand(n), rand(m)
S = zeros(n,n+m+1)
∇fd1(S,x,u,dt)
∇fd1!, = generate_jacobian(fd1,n,m)
@test ∇fd1!(x,u) == S[:,1:n+m]

∇fd1!,fd1_aug! = generate_jacobian(Discrete,fd1,n,m)
∇fd1!(x,u,dt)
model1 = AnalyticalModel{Nominal,Discrete}(fd1,n,m,0)
model2 = AnalyticalModel{Nominal,Discrete}(fd1,∇fd1,n,m,0,check_functions=true)
@test _test_jacobian(Discrete,∇fd1) == [false,true,false]
@test_nowarn _check_jacobian(Discrete,fd1,∇fd1,n,m)

@test model1.∇f(x,u,dt)[:,1:6] == S[:,1:6]
model2.∇f(S,x,u,dt)
@test model1.∇f(x,u,dt)[:,1:n+m] == S[:,1:n+m]
model1.∇f(S,zeros(n),x,u,dt)
t_fd = @elapsed model1.∇f(S,x,u,dt)
t_an = @elapsed model2.∇f(S,x,u,dt)
@test t_an*1.5 < t_fd
model1.f(ẋ,x,u)
model2.f(ẋ2,x,u)
@test ẋ == ẋ2

@inferred model1.∇f(S,x,u,dt)
@inferred model2.∇f(S,x,u,dt)

# Create discrete dynamics from continuous
n,m = 3,2
model = Dynamics.car_model
discretizer = rk3
model_d = Model{Nominal,Discrete}(model,discretizer)

# Test partitioning
Z = BlockMatrix(model)
@test size(Z) == (n,n+m)
@test size(Z.xx) == (n,n)
@test size(Z.xu) == (n,m)
Z = BlockMatrix(Int,model)
@test Z isa BlockMatrix{Int,Matrix{Int}}

S = BlockMatrix(model_d)
@test size(S) == (n,n+m+1)
@test size(S.xx) == (n,n)
@test size(S.xu) == (n,m)
@test size(S.xdt) == (n,1)
S = BlockMatrix(Int,model_d)
@test size(S) == (n,n+m+1)
@test size(S.xx) == (n,n)
@test size(S.xu) == (n,m)
@test size(S.xdt) == (n,1)
@test S isa BlockMatrix{Int,Matrix{Int}}

z = BlockVector(model)
@test length(z) == n+m
s = BlockVector(model_d)
@test length(s) == n+m+1
@test length(model) == n+m
@test length(model_d) == n+m+1

# Generate discrete dynamics equations
f! = model.f
fd! = discretizer(f!, dt)
f_aug! = f_augmented!(f!, n, m)
fd_aug! = discretizer(f_aug!)
nm1 = n + m + 1
In = 1.0*Matrix(I,n,n)

# Initialize discrete and continuous dynamics Jacobians
Jd = zeros(nm1, nm1)
Sd = zeros(nm1)
Sdotd = zero(Sd)
Fd!(Jd,Sdotd,Sd) = ForwardDiff.jacobian!(Jd,fd_aug!,Sdotd,Sd)
Sdotd

function fd_jacobians!(fdx,fdu,x,u)

    # Assign state, control (and dt) to augmented vector
    Sd[1:n] = x
    Sd[n+1:n+m] = u[1:m]
    Sd[end] = √dt

    # Calculate Jacobian
    Fd!(Jd,Sdotd,Sd)

    fdx[1:n,1:n] = Jd[1:n,1:n]
    fdu[1:n,1:m] = Jd[1:n,n.+(1:m)]
end

# Test against previous method
x,u = rand(n),rand(m)
ẋ,ẋ2 = zeros(n), zeros(n)

model_d.f(ẋ,x,u,dt)
fd!(ẋ2,x,u,dt)
@test ẋ == ẋ2
fdx = zeros(model.n,model.n); fdu = zeros(model.n,model.m)
S = zeros(n,nm1)
fd_jacobians!(fdx,fdu,x,u)
@test model_d.∇f(x,u,dt)[:,1:n+m] == [fdx fdu]
@inferred model_d.f(ẋ,x,u,dt)
@inferred model_d.∇f(S,x,u,dt)

S2 = zero(S)
jacobian!(S2,model_d,x,u,dt)
evaluate!(ẋ2,model_d,x,u,dt)
@test ẋ == ẋ2
@test S2 == S

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
