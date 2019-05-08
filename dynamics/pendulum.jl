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


function pendulum_dynamics_uncertain!(ẋ::AbstractVector{T},x::AbstractVector{T},
        u::AbstractVector{T},w::AbstractVector{T}) where T
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    ẋ[1] = x[2]
    ẋ[2] = (u[1] - (m + w[1])*g*lc*sin(x[1]) - b*x[2])/I
end

function pendulum_dynamics_uncertain_backward!(ẋ::AbstractVector{T},x::AbstractVector{T},
        u::AbstractVector{T},w::AbstractVector{T}) where T
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    ẋ[1] = -x[2]
    ẋ[2] = -(u[1] - (m + w[1])*g*lc*sin(x[1]) - b*x[2])/I
end

n,m,r = 2,1,1
pendulum_model_uncertain = UncertainModel(pendulum_dynamics_uncertain!,n,m,r) # inplace model
pendulum_model_uncertain_backward = UncertainModel(pendulum_dynamics_uncertain_backward!,n,m,r) # inplace model
