## Pendulum
# https://github.com/HarvardAgileRoboticsLab/unscented-dynamic-programming/blob/master/pendulum_dynamics.m
function pendulum_dynamics!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    ẋ[1] = x[2]
    ẋ[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/I
end

n,m = 2,1
pendulum_model = Model(pendulum_dynamics!,n,m) # inplace model

# costs
Q = 1.0e-1*Diagonal(I,n)
Qf = 1000.0*Diagonal(I,n)
R = 1.0e-1*Diagonal(I,m)
x0 = [0; 0.]
xf = [pi; 0] # (ie, swing up)

pendulum_costfun = TrajectoryOptimization.LQRCost(Q,R,Qf,xf)
