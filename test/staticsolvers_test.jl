using StaticArrays, LinearAlgebra

max_con_viol = 1.0e-8
verbose=false
prob = copy(Problems.quad_obs)
model = continuous(prob.model)
model_d = rk3(model)
n,m,N = size(prob)
dt = prob.dt
ix = 1:n
iu = n .+ (1:m)
const ix_ = @SVector [i for i = ix]
const iu_ = @SVector [i for i = iu]

# Random inputs
xs,us = (@SVector rand(n)), (@SVector rand(m))
x,u = Array(xs), Array(us)
zs = [xs;us]
z = [x;u]


quad_ = Dynamics.Quadrotor()
dynamics(quad_,x,u)
jacobian(quad_,x,u)
generate_jacobian(quad_)
jacobian(quad_,x,u)

fc(x,u) = Dynamics.quadrotor_dynamics(x,u,Dynamics.quadrotor_params)
∇fc = generate_jacobian_nip(fc, n, m)
∇fc2 = generate_jacobian_nip(fc, ix_, iu_)
quad = SModel{Continuous}(fc,n,m)


# Continuous time
fc(x,u) ≈ dynamics(quad, x, u)
fc(x,u) ≈ dynamics(quad_, x, u)
xdot = zeros(n)
model.f(xdot,x,u)
fc(x,u) ≈ xdot

∇fc(x,u) ≈ jacobian(quad,x,u)
∇fc(x,u) ≈ ∇fc2(x,u)
∇fc(x,u) ≈ jacobian(quad_,x,u)
Z = zeros(n,n+m)
jacobian!(Z,model,x,u)
Z ≈ ∇fc(x,u)

@btime model.f($xdot,$x,$u)
@btime fc($x,$u)
@btime dynamics($quad,$x,$u)
@btime dynamics($quad,$xs,$us)
@btime dynamics($quad_,$xs,$us)

@btime jacobian!($Z,$model,$x,$u)
@btime jacobian($quad,$x,$u)
@btime jacobian($quad,$xs,$us)
@btime jacobian($quad_,$xs,$us)
@btime jacobian($quad_,$zs)


# Discrete Time
quad_d = discretize(quad,:rk3_nip)
f_d2 = rk3_nip(quad_)
f_d = rk3_nip(fc)

f_d(x,u,dt) ≈ dynamics(quad_d,x,u,dt)
model_d.f(xdot,x,u,dt)
xdot ≈ f_d(x,u,dt)
xdot ≈ f_d2(x,u,dt)
f_d(x,u,dt) isa SVector
rk3_gen(quad_)
discrete_dynamics(quad_,xs,us,dt)

S = zeros(n,n+m+1)
model_d.∇f(S,x,u,dt)
S ≈ jacobian(quad_d,x,u,dt)
ss = [xs;us; (@SVector [dt])]
s = [x;u;dt]
generate_discrete_jacobian(quad_)
discrete_jacobian(quad_,xs,us,dt) ≈ discrete_jacobian(quad_,ss) ≈ discrete_jacobian(quad_,zs,dt)

@btime model_d.f($xdot,$x,$u,$dt)
@btime f_d($xs,$us,$dt)
@btime f_d2($xs,$us,$dt)
@btime dynamics($quad_d,$x,$u,$dt)
@btime dynamics($quad_d,$xs,$us,$dt)
@btime discrete_dynamics($quad_,$x,$u,$dt)

@btime model_d.∇f($S,$x,$u,$dt)
@btime jacobian($quad_d,$x,$u,$dt)
@btime jacobian($quad_d,$xs,$us,$dt)
@btime discrete_jacobian($quad_,$xs,$us,$dt)
@btime discrete_jacobian($quad_,$x,$u,$dt)
@btime discrete_jacobian($quad_,$ss) # 0 alloc
@btime discrete_jacobian($quad_,$zs,$dt) # 0 alloc, 3x faster than in place


# Solve the problem using the old method
prob = copy(Problems.quad_obs)
dt = prob.dt
opts_ilqr = iLQRSolverOptions()
r0,s0 = solve(prob,opts_ilqr)
Dt = ones(N)*dt
X,U = r0.X, r0.U

# Static Objective
q0 = [1.;0.;0.;0.] # unit quaternion

x0 = zeros(n)
x0[1:3] = [0.; 0.; 10.]
x0[4:7] = q0

xf = zero(x0)
xf[1:3] = [0.;60.; 10.]
xf[4:7] = q0;

# cost
Q = (1.0e-3)*Diagonal(@SVector ones(n))
R = (1.0e-2)*Diagonal(@SVector ones(m))
Qf = 1.0*Diagonal(@SVector ones(n))
obj = LQRObjective(Q, R, Qf, xf, N) # objective with same stagewise costs
mycost = obj[1]
Qx = mycost.Q*x
Qu = mycost.R*u
Qxx = mycost.Q
Quu = mycost.R
Qux = mycost.H
E = StaticExpansion(Qx,Qu,Qxx,Quu,Qux)

mycost.Q*xs
mycost.Q*x
@which StaticExpansion(mycost.Q*x + mycost.q, mycost.R*u + mycost.r, mycost.Q, mycost.R, mycost.H)
@btime cost($prob.obj, $r0.X, $r0.U, get_dt_traj($prob))
@btime cost($obj, $r0.X, $r0.U, get_dt_traj($prob))  # much faster

# Cost Expansions
E = StaticExpansion{Float64}(n,m,N)


opts = StaticiLQRSolverOptions()
silqr = StaticiLQRSolver(prob,opts)
silqr.S[1].x
cost_expansion!(r0, silqr)
cost_expansion!(r0, s0)
@btime cost_expansion!($r0, $s0)
@btime cost_expansion!($r0, $silqr) # 2x faster



# Test Jacobians
prob_ = update_problem(prob, obj=obj)
Zs = [@SMatrix zeros(n,n+m+1) for k = 1:N-1]
Z = s0.∇F
Dt = ones(N)*dt
X,U = r0.X, r0.U
jacobian!(Zs,quad_d,X,U,Dt)
jacobian!(Z,model_d,X,U,Dt)


@btime jacobian!($Z,$model_d,$X,$U,$Dt)
@btime jacobian!($Zs,$quad_d,$X,$U,$Dt)



# iLQR
ilqr = s0
opts = StaticiLQRSolverOptions()
silqr = StaticiLQRSolver(prob, opts)
jacobian!(silqr.∇F, quad_d, X, U, Dt)
cost_expansion!(r0, silqr)
Juno.@enter backwardpass!(prob, silqr)

J_prev = cost(r0)

jacobian!(silqr.∇F, quad_d, X, U, Dt)
jacobian!(r0, ilqr)
silqr.∇F ≈ s0.∇F

cost_expansion!(r0, silqr)
cost_expansion!(r0, ilqr)
all([silqr.Q[k].xx == ilqr.Q[k].xx for k = 1:N])
all([silqr.Q[k].uu == ilqr.Q[k].uu for k = 1:N-1])
all([silqr.Q[k].ux == ilqr.Q[k].ux for k = 1:N-1])
all([silqr.Q[k].x == ilqr.Q[k].x for k = 1:N])
all([silqr.Q[k].u == ilqr.Q[k].u for k = 1:N-1])

backwardpass!(r0, silqr) ≈ backwardpass!(r0, ilqr)

cost_expansion!(r0, silqr)
∇V = backwardpass!(r0, silqr)
forwardpass!(r0, ilqr, ∇V, J_prev)
forwardpass!(r0, silqr, ∇V, J_prev)



for k = 1:N
    silqr.Q.xx[k]

@btime begin
    cost_expansion!($r0, $silqr)
    backwardpass!($prob, $silqr)
end
@btime begin
    cost_expansion!($r0, $s0)
    backwardpass!($prob, $s0)
end

@which cost(prob.obj, prob.X, prob.U, get_dt_traj(prob))

Expansion(n,m)
