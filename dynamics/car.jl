# Car

function car_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector)
    ẋ[1] = u[1]*cos(x[3])
    ẋ[2] = u[1]*sin(x[3])
    ẋ[3] = u[2]
    return nothing
end
n,m = 3,2

car = Model(car_dynamics!,n,m)
