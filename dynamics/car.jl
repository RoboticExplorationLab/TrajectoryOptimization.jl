# # Car
#
# function car_dynamics!(ẋ::AbstractVector,x::AbstractVector,u::AbstractVector)
#     ẋ[1] = u[1]*cos(x[3])
#     ẋ[2] = u[1]*sin(x[3])
#     ẋ[3] = u[2]
#     return nothing
# end
# n,m = 3,2
#
# car = Model(car_dynamics!,n,m)

@with_kw struct DubinsCar <: AbstractModel
    radius::Float64 = 0.175  # [m] radius of a Roomba
end
Base.size(::DubinsCar) = 3,2

function dynamics(::DubinsCar,x,u)
    ẋ = @SVector [u[1]*cos(x[3]),
                  u[1]*sin(x[3]),
                  u[2]]
end

Base.position(::DubinsCar, x::SVector) = @SVector [x[1], x[2], 0.0]
orientation(::DubinsCar, x::SVector) = expm(x[3]*@SVector [0,0,1.])

# using Plots
# function Plots.plot(model::DubinsCar, Z::Traj; kwargs...)
#     x = [z.z[1] for z in Z]
#     y = [z.z[2] for z in Z]
#     Plots.plot(x, y, xlabel="x position", ylabel="y position", markershape=:circle;
#         kwargs...)
# end
# function Plots.plot!(model::DubinsCar, Z::Traj; kwargs...)
#     x = [z.z[1] for z in Z]
#     y = [z.z[2] for z in Z]
#     Plots.plot!(x, y, xlabel="x position", ylabel="y position", markershape=:circle;
#         kwargs...)
# end
