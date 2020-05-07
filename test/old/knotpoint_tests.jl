using Test
using StaticArrays

model = Dynamics.DubinsCar()
x,u = rand(model)
n,m = size(model)
N = 101
dt = 0.1
tf = 10.0

X = [@SVector rand(n) for k = 1:N]
U = [@SVector rand(m) for k = 1:N-1]
Z = Traj(x,u,dt,N)
@test Z[1].dt == dt
@test Z[1].t == 0
@test Z[end].dt == 0
@test Z[end].t == tf

Z = Traj(X,U,fill(dt,N))
@test Z[1].dt == dt
@test Z[1].t == 0
@test Z[end].dt == 0
@test Z[end].t ≈ tf
