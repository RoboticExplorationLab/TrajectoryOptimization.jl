# Tests
using Pendulum
x0 = [1,2]
u = 1
dt = 0.1

dyn_mid = iLQR.f_midpoint(dynamics, 0.1)
fc(x0, u)
dyn_mid(x0,u)

Pendulum.dynamics_midpoint(x0, u, dt) ≈ dyn_mid(x0, u)

println(fc(x0,u))
println(dynamics_midpoint(x0,u))

fx(x0,dt)
fu(x0,dt)
