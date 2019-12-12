prob = copy(Problems.quadrotor)
sprob = copy(Problems.quadrotor_static)
prob = copy(Problems.cartpole)
sprob = copy(Problems.cartpole_static)
U0 = deepcopy(controls(sprob))
X0 = deepcopy(states(sprob))

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

n,m,N = size(silqr)

rollout!(prob)
rollout!(silqr)
states(silqr) ≈ prob.X
controls(silqr) ≈ prob.U

cost(prob.obj, prob.X, prob.U, get_dt_traj(prob)) ≈ cost(silqr)
J_prev = cost(silqr)

jacobian!(prob, ilqr)
discrete_jacobian!(silqr.∇F, silqr.model, silqr.Z)
silqr.∇F ≈ ilqr.∇F
all(map(1:N-1) do k
    A,B = dynamics_expansion(ilqr.∇F[k], prob.model, prob.X[k], prob.U[k])
    As,Bs = dynamics_expansion(silqr.model, silqr.Z[k])
    A ≈ As && B ≈ Bs
end)

cost_expansion!(prob,ilqr)
cost_expansion(silqr.Q, silqr.obj, silqr.Z)

ΔV =  backwardpass!(prob, ilqr)
ΔVs = backwardpass!(silqr)
ilqr.K ≈ silqr.K
ilqr.d ≈ silqr.d
ΔV ≈ ΔVs

ilqr.K[1] ≈ silqr.K[1]
ilqr.d[1] ≈ silqr.d[1]
δx = state_diff(prob.model, ilqr.X̄[1], prob.X[1])
δxs = state_diff(silqr.model, state(silqr.Z̄[1]), state(silqr.Z[1]))
state(silqr.Z̄[1])
state(silqr.Z[1])

ilqr.Ū[1]
control(silqr.Z̄[1])
state_diff(prob.model, ilqr.X̄[2], prob.X[2])
forwardpass!(prob,ilqr, ΔV, J_prev)
forwardpass!(silqr, ΔV, J_prev)

rollout!(prob,ilqr,1.0)
rollout!(silqr,1.0)
cost(prob.obj, ilqr.X̄, ilqr.Ū, get_dt_traj(prob,ilqr.Ū))
cost(silqr, silqr.Z̄)

J = step!(prob, ilqr, J_prev)
Js = step!(silqr, J_prev)
@show J ≈ Js

copyto!(prob.X, ilqr.X̄)
copyto!(prob.U, ilqr.Ū)
for k = 1:N
    silqr.Z[k].z = silqr.Z̄[k].z
end

states(silqr) ≈ prob.X
controls(silqr) ≈ prob.U


prob = copy(Problems.quadrotor)
has_quat(prob.model)
sprob = copy(Problems.quadrotor_static)
U0 = deepcopy(controls(sprob))
X0 = deepcopy(states(sprob))

ilqr = iLQRSolver(prob)
silqr = StaticiLQRSolver(sprob)

@btime begin
    initial_controls!($prob, $U0)
    solve!($prob, $ilqr)
end
ilqr.stats[:iterations]

@btime begin
    initial_controls!($silqr, $U0)
    solve!($silqr)  # 7.8x faster
end
silqr.stats.iterations



@btime quat_diff_jacobian($q1)
@btime state_diff_jacobian($sprob.model, $x0)
include("../dynamics/quaternions.jl")
q1 = normalize(@SVector rand(4))
q2 = normalize(@SVector rand(4))
dq1 = quat_diff(q1,q2)
dq2 = vec(inv(Quaternion(q2))* Quaternion(q1))
dq1 ≈ dq2

q0 = q2
dq(q) = quat_diff(q,q0)
ForwardDiff.jacobian(dq,q1) ≈ quat_diff_jacobian(q0)
model = Dynamics.Quadrotor()
x,u = rand(model)
z = KnotPoint(x,u,prob.dt)
dynamics_expansion(model, z)
@btime dynamics_expansion($model, $z)
@btime discrete_jacobian($model, $z)
