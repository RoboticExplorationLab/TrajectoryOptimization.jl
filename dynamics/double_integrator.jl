# 1D
function double_integrator_dynamics!(ẋ::AbstractVector{T},x::AbstractVector{T},u::AbstractVector{T}) where T
    ẋ[1] = x[2]
    ẋ[2] = u[1]
end

n = 2
m = 1

doubleintegrator = Model(double_integrator_dynamics!,n,m)

# 3D
function double_integrator_3D_dynamics!(ẋ,x,u) where T
    ẋ[1:3] = x[4:6]
    ẋ[4:6] = u[1:3]
    ẋ[6] -= 9.81 # gravity
end

doubleintegrator3D = Model(double_integrator_3D_dynamics!,6,3)
