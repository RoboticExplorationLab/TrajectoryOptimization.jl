# Car

function car_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector)
    ẋ[1] = u[1]*cos(x[3])
    ẋ[2] = u[1]*sin(x[3])
    ẋ[3] = u[2]
    return nothing
end
n,m = 3,2

car_model = Model(car_dynamics!,n,m)

function car_dynamics_uncertain!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector,w::AbstractVector)
    ẋ[1] = u[1]*cos(x[3]) + w[1]
    ẋ[2] = u[1]*sin(x[3]) + w[2]
    ẋ[3] = u[2]*w[3]
    return nothing
end
n,m,r = 3,2,3

car_model_uncertain = UncertainModel(car_dynamics_uncertain!,n,m,r)
