function double_integrator_dynamics!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
    ẋ[1] = x[2]
    ẋ[2] = u[1]
end

n = 2
m = 1

doubleintegrator_model = Model(double_integrator_dynamics!,n,m)
