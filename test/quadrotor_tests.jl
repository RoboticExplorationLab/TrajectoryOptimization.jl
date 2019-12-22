using BenchmarkTools
model = Dynamics.Quadrotor()
x,u = rand(model)
dynamics(model,x,u)
@btime dynamics($model,$x,$u)

model2 = Dynamics.Quadrotor2{UnitQuaternion{Float64,VectorPart}}()
dynamics(model2,x,u) ≈ dynamics(model,x,u)

@btime dynamics($model,$x,$u)
@btime dynamics($model2,$x,$u)
