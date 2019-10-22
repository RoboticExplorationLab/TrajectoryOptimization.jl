using LabelledArrays

prob = copy(Problems.quad_obs)
n,m,N = size(prob)
dt = prob.dt

# Generate model
quad_ = Dynamics.Quadrotor()
generate_jacobian(quad_)
rk3_gen(quad_)
generate_discrete_jacobian(quad_)


# Test model
xs,us = (@SVector rand(n)), (@SVector rand(m))
x,u = Array(xs), Array(us)
zs = [xs;us]
z = [x;u]
initial_controls!(prob, [u for k = 1:N-1])
initial_states!(prob, [x for k = 1:N])

@btime dynamics($quad_,$xs,$us)
@btime evaluate!($x, $prob.model, $x, $u, $dt)
@btime discrete_dynamics($quad_,$xs,$us,$dt) # 11x faster


@btime jacobian($quad_,$zs)
@btime discrete_jacobian($quad_,$zs,$dt)

# Build Trajectory
z = KnotPoint(xs,us,dt)
@btime state($z)
@btime control($z)
@btime is_terminal($z)
Z = [z for k = 1:N]
Z[end] = KnotPoint(xs,m)


# Build Problem
x0 = SVector{n}(prob.x0)
xf = SVector{n}(prob.xf)
Q = (1.0e-3)*Diagonal(@SVector ones(n))  # use static diagonal to avoid allocations
R = (1.0e-2)*Diagonal(@SVector ones(m))
Qf = 1.0*Diagonal(@SVector ones(n))
obj = LQRObjective(Q,R,Qf,xf,N)

sprob = StaticProblem(quad_, obj, x0, xf, Z, N, dt, prob.tf)

# Build Solver
silqr = StaticiLQRSolver(prob)
ilqr = iLQRSolver(prob)


# Dynamics evaluations
dyn = [xs for k = 1:N]
discrete_dynamics(quad_, z)
Ix = SMatrix{n+m,n}( Matrix(I,n+m,n) )
@btime propagate_dynamics($quad_, $Z[2], $Z[1], $Ix)

@btime discrete_dynamics!($dyn, $quad_, $Z)


# Dynamics jacobians
∇f = silqr.∇F
jacobian!(prob, ilqr)
discrete_jacobian!(∇f, quad_, Z)
silqr.∇F ≈ ilqr.∇F
@btime jacobian!($prob, $ilqr)
@btime discrete_jacobian!($∇f, $quad_, $Z) # ≈3x faster

# Objective
cost(prob) ≈ cost(obj, Z)
@btime cost($prob)
@btime cost($obj, $Z) # 7.5x faster


# Cost expansion
E = silqr.Q
cost_expansion!(prob, ilqr)
cost_expansion(E, obj, Z)
all([E.xx[k] ≈ ilqr.Q[k].xx for k in eachindex(E.xx)])
all([E.uu[k] ≈ ilqr.Q[k].uu for k = 1:N-1])
all([E.x[k] ≈ ilqr.Q[k].x for k in eachindex(E.x)])
all([E.u[k] ≈ ilqr.Q[k].u for k = 1:N-1])

@btime cost_expansion!($prob, $ilqr)
@btime cost_expansion($E, $obj, $Z)  # 7x faster


# Rollout
x = zeros(n)*NaN
u = ones(m)
Z = [KnotPoint(x,u,dt) for k = 1:N]
Z[N] = KnotPoint(x,m)


prob = copy(Problems.quad_obs)
initial_controls!(prob, [u for k = 1:N-1])
sprob = StaticProblem(quad_, obj, xf, x0, Z, N, dt, prob.tf)
rollout!(sprob)
rollout!(prob)
all([state(sprob.Z[k]) ≈ prob.X[k] for k = 1:N])

@btime rollout!($prob)
@btime rollout!($sprob) # 11x faster


# Backward pass
silqr = StaticiLQRSolver(prob)
ilqr = iLQRSolver(prob)
cost_expansion!(prob, ilqr)
cost_expansion(silqr.Q, sprob.obj, sprob.Z)
jacobian!(prob, ilqr)
discrete_jacobian!(silqr.∇F, quad_, Z)
backwardpass!(prob, ilqr)
backwardpass!(sprob, silqr)
@btime backwardpass!($prob, $ilqr)
@btime backwardpass!($sprob, $silqr) # 11x speedup

using InteractiveUtils
@btime _cost_to_go($Qk, $silqr.K[N-1], $silqr.d[N-1])
silqr.K ≈ ilqr.K
silqr.d ≈ ilqr.d




Quu_reg = @SMatrix zeros(5,5)
Quu_reg += I(5)

cholesky(Matrix(Quu_reg), check=false)
@btime cholesky($Quu_reg)
@btime cholesky(Matrix($Quu_reg), check=false)
cholesky(Array(Quu_reg), check=false)
isposdef(Quu_reg)

silqr.opts.bp_reg_type

silqr.∇F[1]


silqr.S.xx

function getQ(Q,k)
    return (x=Q.x[k], u=Q.u[k], xx=Q.xx[k], uu=Q.uu[k], ux=Q.ux[k])
end
Qk = getQ(silqr.Q,1)
Qk.x = @SVector ones(13)
@btime getQ($silqr.Q, 1)
@btime $silqr.Q[1]

mychol(Size(Quu_reg), Quu_reg)
LAPACK.potrf!('U', Quu_reg)
