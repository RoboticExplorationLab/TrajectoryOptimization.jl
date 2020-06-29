@testset "Trajectories" begin
n,m,N = 5,3,11
dt = 0.1
x = @SVector rand(n)
u = @SVector rand(m)
z = KnotPoint(x,u,dt)

#--- Empty contructor
Z = Traj(n,m,dt,N)
@test all(isnan, state(Z[rand(1:N)]))
@test control(Z[rand(1:N-1)]) == zeros(m)
@test length(controls(Z)) == N-1
@test length(states(Z)) == N
@test eltype(states(Z)) <: SVector{n}
@test eltype(controls(Z)) <: SVector{m}
@test Z[1].dt ≈ dt
@test Z[end].dt ≈ 0
@test Z[1].t ≈ 0
@test Z[N].t ≈ (N-1)*dt
@test TO.is_terminal(Z[end]) == true

Z = Traj(n,m,dt,N, equal=true)
@test all(isnan, state(Z[rand(1:N)]))
@test control(Z[rand(1:N)]) == zeros(m)
@test length(controls(Z)) == N
@test length(states(Z)) == N
@test eltype(states(Z)) <: SVector{n}
@test eltype(controls(Z)) <: SVector{m}
@test Z[1].dt ≈ dt
@test Z[end].dt ≈ dt
@test Z[1].t ≈ 0
@test Z[N].t ≈ (N-1)*dt
@test TO.is_terminal(Z[end]) == false

#--- Copy single constructor
Z = Traj(x, u, dt, N)
@test length(Z) == N
@test eltype(Z) <: KnotPoint{Float64,n,m}
@test state(Z[1]) == x
Z[1].z = 2*Z[1].z
@test state(Z[1]) ≈ 2x
@test state(Z[2]) ≈ x
@test control(Z[1]) ≈ 2u
@test control(Z[2]) ≈ u
@test Z[1].dt ≈ dt
@test Z[end].dt ≈ 0
@test Z[1].t ≈ 0
@test Z[N].t ≈ (N-1)*dt

#--- Vector constructor
X = [@SVector rand(n) for k = 1:N]
U = [@SVector rand(m) for k = 1:N-1]
Z = Traj(X,U,fill(dt,N))
@test states(Z) ≈ X
@test controls(Z) ≈ U
@test TO.get_times(Z) ≈ range(0, length=N, step=dt)

Z2 = copy(Z)
RobotDynamics.set_state!(Z[1], x)
@test !(state(Z[1]) ≈ state(Z2[1]))
@test state(Z[2]) ≈ state(Z2[2])
X = 2 .* X
U = 3 .* U
X[1] = x
TO.set_states!(Z2, X)
@test state(Z[1]) ≈ state(Z2[1])
@test !(control(Z2[2]) ≈ U[2])
TO.set_controls!(Z2, U)
@test control(Z2[2]) ≈ U[2]

@test TO.is_terminal(Z[end]) == true
push!(U, 2*U[end])
Z = Traj(X,U,fill(dt,N))
@test length(controls(Z)) == N
@test TO.is_terminal(Z[end]) == false

#--- Test copyto!
Z0 = [KnotPoint(3*X[k], 2*U[k], dt, dt*(k-1)) for k = 1:N]
@test state.(Z0) ≈ 3 .* states(Z)
@test control.(Z0) ≈ 2 .* controls(Z)
copyto!(Z0, Z)
@test state.(Z0) ≈ states(Z)
@test control.(Z0) ≈ controls(Z)

Z2 = Traj(rand() .* X, rand() .* U,fill(dt,N))
@test !(states(Z) ≈ states(Z2))
@test !(controls(Z) ≈ controls(Z2))
copyto!(Z2, Z)
@test states(Z) ≈ states(Z2)
@test controls(Z) ≈ controls(Z2)

#--- Test functions on trajectories
model = Cartpole()
n,m = size(model)
fVal = [@SVector zeros(n) for k = 1:N]
X = [@SVector rand(n) for k = 1:N]
U = [@SVector rand(m) for k = 1:N-1]
Z = Traj(X,U,fill(dt,N))

# Test dynamics evaluation
TO.discrete_dynamics!(RK3, fVal, model, Z)
@test fVal[1] ≈ discrete_dynamics(RK3, model, X[1], U[1], 0.0, dt)
dyn = TO.DynamicsConstraint{RK3}(model, N)
conval = TO.ConVal(n,m, dyn, 1:N-1)
TO.evaluate!(conval, Z)
@test conval.vals ≈ fVal[1:N-1] .- X[2:N]
@test !(conval.vals ≈ [zeros(n) for k = 1:N-1])

# Test rollout
rollout!(model, Z, X[1])
TO.evaluate!(conval, Z)
@test conval.vals ≈ [zeros(n) for k = 1:N-1]
end
