function double_integrator_dynamics!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
    ẋ[1] = x[2]
    ẋ[2] = u[1]
end

n = 2
m = 1

doubleintegrator_model = Model(double_integrator_dynamics!,n,m)

function double_integrator_dynamics_uncertain!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T},w::AbstractVector{T}) where T
    ẋ[1] = (x[2] + w[1])
    ẋ[2] = u[1]
end

n = 2
m = 1
r = 1

doubleintegrator_model_uncertain = UncertainModel(double_integrator_dynamics_uncertain!,n,m,r)
