using StaticArrays, LinearAlgebra

max_con_viol = 1.0e-8
verbose=false
prob = copy(Problems.quad_obs)
model = continuous(prob.model)
model_d = rk3(model)
n,m,N = size(prob)
dt = prob.dt

params = Dynamics.quad_params

fc(x,u) = Dynamics.quadrotor_dynamics(x,u,params)
∇fc = generate_jacobian_nip(fc,n,m)
quad = SModel{Continuous}(fc,n,m)

# Random inputs
xs,us = (@SVector rand(n)), (@SVector rand(m))
x,u = Array(xs), Array(us)

# Continuous time
fc(x,u) ≈ dynamics(quad, x, u)
xdot = zeros(n)
model.f(xdot,x,u)
fc(x,u) ≈ xdot

∇fc(x,u) ≈ jacobian(quad,x,u)
Z = zeros(n,n+m)
jacobian!(Z,model,x,u)
Z ≈ ∇fc(x,u)

@btime model.f($xdot,$x,$u)
@btime fc($x,$u)
@btime dynamics($quad,$x,$u)
@btime dynamics($quad,$xs,$us)

@btime jacobian!($Z,$model,$x,$u)
@btime ∇fc($x,$u)
@btime jacobian($quad,$x,$u)
@btime jacobian($quad,$xs,$us)


# Discrete Time
quad_d = discretize(quad,:rk3_nip)
fd = rk3_nip(fc)

fd(x,u,dt) ≈ dynamics(quad_d,x,u,dt)
model_d.f(xdot,x,u,dt)
xdot ≈ fd(x,u,dt)
fd(x,u,dt) isa SVector

ix = @SVector [i for i = 1:n]
iu = @SVector [i for i = 1:m]
f_aug(z) = fd(z[ix],z[iu],z[end])
f_aug([x;u;dt]) isa SVector
ForwardDiff.jacobian(f_aug,[xs;us;@SVector [dt,]])

∇fd = generate_jacobian_nip(fd,n,m,dt)
∇fd(x,u,dt) ≈ jacobian(quad_d,x,u,dt)
S = zeros(n,n+m+1)
model_d.∇f(S,x,u,dt)
S ≈ ∇fd(x,u,dt)
∇fd(x,u,dt) isa SMatrix
S = ∇fd(xs,us,dt)
S  isa SMatrix

ix = @SVector [i for i = 1:n]
iu = @SVector [i for i = 1:m]

jacobian(quad_d,xs,us,dt)
S = ∇fd(x,u,dt)
S isa SMatrix  # TODO: fix this

@btime model_d.f($xdot,$x,$u,$dt)
@btime fd($x,$u,$dt)
@btime dynamics($quad_d,$x,$u,$dt)
@btime dynamics($quad_d,$xs,$us,$dt)

@btime model_d.∇f($S,$x,$u,$dt)
@btime ∇fd($x,$u,$dt)
@btime ∇fd($xs,$us,$dt)
@btime jacobian($quad_d,$x,$u,$dt)
@btime jacobian($quad_d,$xs,$us,$dt)


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
@btime backwardpass!($r0, $ilqr)
@btime backwardpass!($r0, $silqr)

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
