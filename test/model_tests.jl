
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
Z = PartedMatrix(model)
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
Z1 = zero(Z)
Z2 = zero(Z)
model1.∇f(Z1,x,u);
model2.∇f(Z2,x,u);
@test Z1 == Z2
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
TO._check_dynamics(f3_p,n,m)
model3 = Model(f3,n,m,params)
model4 = Model(f3,∇f3,n,m,params)
model3.∇f(Z1,x,u); model4.∇f(Z2,x,u);
@test Z1 == Z2
@test Z1 == Z
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

∇fd1!,fd1_aug! = TO.generate_jacobian(Nominal,Discrete,fd1,n,m)
∇fd1!(x,u,dt)
model1 = AnalyticalModel{Nominal,Discrete}(fd1,n,m,0)
model2 = AnalyticalModel{Nominal,Discrete}(fd1,∇fd1,n,m,0, check_functions=true)
@test TO._test_jacobian(Discrete,∇fd1) == [false,true,false]
# @test_nowarn _check_jacobian(Discrete,fd1,∇fd1,n,m)

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
discretizer = :rk3
model_d = discretize_model(model,discretizer)

# Test partitioning
Z = PartedMatrix(model)
@test size(Z) == (n,n+m)
@test size(Z.xx) == (n,n)
@test size(Z.xu) == (n,m)
Z = PartedMatrix(Int,model)
@test Z isa PartedMatrix{Int,Matrix{Int}}

S = PartedMatrix(model_d)
@test size(S) == (n,n+m+1)
@test size(S.xx) == (n,n)
@test size(S.xu) == (n,m)
@test size(S.xdt) == (n,1)
S = PartedMatrix(Int,model_d)
@test size(S) == (n,n+m+1)
@test size(S.xx) == (n,n)
@test size(S.xu) == (n,m)
@test size(S.xdt) == (n,1)
@test S isa PartedMatrix{Int,Matrix{Int}}

z = PartedVector(model)
@test length(z) == n+m
s = PartedVector(model_d)
@test length(s) == n+m+1
@test length(model) == n+m
@test length(model_d) == n+m+1

# Generate discrete dynamics equations
nm1 = n + m + 1
In = 1.0*Matrix(I,n,n)
x,u = rand(n), rand(m)

model_d.f(ẋ,x,u,dt)
fdx = zeros(model.n,model.n); fdu = zeros(model.n,model.m)
S = zeros(n,nm1)
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
