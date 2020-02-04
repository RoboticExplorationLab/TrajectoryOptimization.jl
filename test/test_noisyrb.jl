using Distributions
using Test
using LinearAlgebra
model = Dynamics.Quadrotor2()
x,u = rand(model)

w0 = MvNormal(Diagonal(ones(6))*0)
model0 = Dynamics.NoisyRB(model, w0)
@test dynamics(model, x, u) ≈ dynamics(model0,x,u)

w1 = MvNormal(Diagonal(ones(6))*1)
model1 = Dynamics.NoisyRB(model, w1)
@test !(dynamics(model, x, u) ≈ dynamics(model1,x,u))
