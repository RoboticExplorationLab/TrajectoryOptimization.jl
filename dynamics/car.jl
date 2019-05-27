# Car

function car_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector)
    ẋ[1] = u[1]*cos(x[3])
    ẋ[2] = u[1]*sin(x[3])
    ẋ[3] = u[2]
    return nothing
end
n,m = 3,2

car_model = Model(car_dynamics!,n,m)

Q = (1e-2)*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = (1e-2)*Diagonal(I,m)
x0 = [0.;0.;0.]
xf = [0.;1.;0.]
dt = 0.01

car_costfun = TrajectoryOptimization.LQRCost(Q, R, Qf, xf)
