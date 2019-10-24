
# Original problem
prob = copy(Problems.quad_obs)
n,m,N = size(prob)
dt = prob.dt

# Generate model
quad_ = Dynamics.Quadrotor()
generate_jacobian(quad_)
rk3_gen(quad_)
generate_discrete_jacobian(quad_)


# Static Objective
x0 = SVector{n}(prob.x0)
xf = SVector{n}(prob.xf)
Q = (1.0e-3)*Diagonal(@SVector ones(n))  # use static diagonal to avoid allocations
R = (1.0e-2)*Diagonal(@SVector ones(m))
Qf = 1.0*Diagonal(@SVector ones(n))
obj = LQRObjective(Q,R,Qf,xf,N)


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
sprob = StaticProblem(quad_, obj, x0, xf, Z, deepcopy(Z), N, dt, prob.tf)

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
@btime cost($(sprob.obj), $Z) # 7.5x faster


# Cost expansion
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
sprob = StaticProblem(quad_, obj, x0, xf, Z, deepcopy(Z), N, dt, prob.tf)
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
ΔV = backwardpass!(sprob, silqr)

silqr.K ≈ ilqr.K
silqr.d ≈ ilqr.d

@btime backwardpass!($prob, $ilqr)
@btime backwardpass!($sprob, $silqr) # 11x speedup



# Rollout
rollout!(prob, ilqr, 1.0)
rollout!(sprob, silqr, 1.0)
all([state(sprob.Z̄[k]) ≈ ilqr.X̄[k] for k in eachindex(sprob.Z)])
all([state(sprob.Z[k]) ≈ prob.X[k] for k in eachindex(sprob.Z)])
all([control(sprob.Z[k]) ≈ prob.U[k] for k in eachindex(prob.U)])

@btime rollout!($prob, $ilqr, 1.0)
@btime rollout!($sprob, $silqr, 1.0)  # 8.5x speedup

# Forward pass
_J = zeros(N)
cost!(_J, sprob.obj, sprob.Z)
J_prev = sum(_J)
forwardpass!(prob, ilqr, Vector(ΔV), J_prev)
forwardpass!(sprob, silqr, ΔV, J_prev, _J)
all([state(sprob.Z̄[k]) ≈ ilqr.X̄[k] for k in eachindex(sprob.Z)])
all([control(sprob.Z̄[k]) ≈ ilqr.Ū[k] for k in 1:N-1])
all([state(sprob.Z[k]) ≈ prob.X[k] for k in 1:N-1])

@btime forwardpass!($prob, $ilqr, $(Vector(ΔV)), $J_prev)
@btime forwardpass!($sprob, $silqr, $ΔV, $J_prev, $_J) # 8.5x speedup

# Test Entire Step
x = zeros(n)*NaN
u = ones(m)
Z = [KnotPoint(x,u,dt) for k = 1:N]
Z[N] = KnotPoint(x,m)

prob = copy(Problems.quad_obs)
initial_controls!(prob, [u for k = 1:N-1])
sprob = StaticProblem(quad_, obj, x0, xf, Z, deepcopy(Z), N, dt, prob.tf)

rollout!(sprob)
rollout!(prob)
J_prev = cost(prob)
J_prev == cost(sprob.obj, sprob.Z)

step!(sprob, silqr, J_prev)
step!(prob, ilqr, J_prev)

@btime step!($prob, $ilqr, $J_prev)
@btime step!($sprob, $silqr, $J_prev) # 9.5x faster

gradient_todorov!(sprob, silqr)
mean(silqr.grad) ≈ gradient_todorov(prob, ilqr)

@btime gradient_todorov($prob, $ilqr)
@btime gradient_todorov!($sprob, $silqr, $grad) # 3.5x faster


# Test entire solve
x = zeros(n)*NaN
u = ones(m)
U0 = [u for k = 1:N-1]
Z = [KnotPoint(x,u,dt) for k = 1:N]
Z[N] = KnotPoint(x,m)

prob = copy(Problems.quad_obs)
initial_controls!(prob, U0)
sprob = StaticProblem(quad_, obj, x0, xf, deepcopy(Z), deepcopy(Z), N, dt, prob.tf)

silqr = StaticiLQRSolver(sprob)
ilqr = iLQRSolver(prob)

solve!(prob, ilqr)
solve!(sprob, silqr)
norm(state(sprob) - prob.X)


@btime begin
    initial_controls!($prob, $U0)
    solve!($prob, $ilqr)
end
ilqr.stats[:iterations]

@btime begin
    for k = 1:$N
        $sprob.Z[k].z = $Z[k].z
    end
    solve!($sprob, $silqr)  # 13x faster!!! (0 allocs)
end
silqr.stats.iterations
