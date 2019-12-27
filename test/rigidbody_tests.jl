using Test

model_q1 = Dynamics.FreeBody{UnitQuaternion{Float64,VectorPart},Float64}()
model_q2 = Dynamics.FreeBody{UnitQuaternion{Float64,ExponentialMap},Float64}()
model_q3 = Dynamics.FreeBody{UnitQuaternion{Float64,ModifiedRodriguesParam},Float64}()
model_p = Dynamics.FreeBody{MRP{Float64},Float64}()
model_e = Dynamics.FreeBody{RPY{Float64},Float64}()

x,u = rand(model_p)

model_p isa Dynamics.FreeBody{<:Rotation}
size(model_e)
size(model_q1)

rand(MRP{Float64})
rand(RPY{Float64})
typeof(MRP{Float64})
D = Dynamics.rotation_type(model_q1)
rand(D)
