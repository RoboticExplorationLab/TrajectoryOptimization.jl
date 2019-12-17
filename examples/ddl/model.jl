using Parameters, StaticArrays, BenchmarkTools, LinearAlgebra
using MAT
using TrajectoryOptimization
import TrajectoryOptimization: dynamics



@with_kw mutable struct BicycleCar{T} <: AbstractModel
    k_s::Dict{Float64,Float64} =
        Dict( collect(range(0,stop=30, length=100)) .=> fill(0.1, 100) )
    a::T = 1.2169  # dist to front axle (m)
    b::T = 1.4131  # dist to rear axle (m)
    h::T = 0.5       # height of center of gravity? (m)
    mass::T = 1744.  # mass (kg)
    Iz::T = 2.7635e3  # moment of inertia (kg⋅m²)
    Cαf::T = 66000. # (N/rad)
    Cαr::T = 60000. # (N/rad)
    μ::T = 0.25  # coefficient of friction
    Cd0::T = 218.06   # Cd0 (N)
    Cd1::T = 0.0 # (N/mps)
    g::T = 9.81  # gravity (m/s²)
end

labels(::BicycleCar) = ["delta","fx","r","Uy","Ux","dpsi","e","t"]

Base.size(::BicycleCar) = 8,2

function dynamics(car::BicycleCar, x, u, s)
    δ_dot  = u[1]  # steering angle rate
    fx_dot = u[2]  # accel rate
    δ  = x[1]  # steering angle
    fx = x[2]  # acceleration
    r  = x[3]  # yaw rate
    Uy = x[4]  # lateral velocity
    Ux = x[5]  # longitudinal velocity
    Δψ = x[6]  # heading error
    e  = x[7]  # lateral error
    t  = x[8]  # time

    # Road curvature
    k = car.k_s[s]

    # Drag Force
    Fx_drag = 0.0

    # Spatial derivative
    s_dot = (Ux * cos(Δψ) - Uy * sin(Δψ)) / (1 - k*e)

    # Get tire forces
    Fxf, Fxr, Fyf, Fyr = logit_lateral_force_model(car, x)

    # State Derivatives
    r_dot  = (car.a * (Fxf * cos(δ) + Fxf*sin(δ)) - car.b * Fyr) / car.Iz
    Ux_dot =  r * Uy + (Fxf * cos(δ) - Fyf * sin(δ) + Fxr - Fx_drag) / car.mass
    Uy_dot = -r * Ux + (Fyf * cos(δ) + Fxf * sin(δ)) / car.mass
    Δψ_dot = r - k * s_dot
    e_dot = Ux * sin(Δψ) + Uy * cos(Δψ)
    t_dot = 1 / s_dot

    return @SVector [δ_dot, fx_dot, r_dot, Uy_dot, Ux_dot, Δψ_dot, e_dot, t_dot]
end

function FWD_force_model(car::BicycleCar, fx)
    l = car.a + car.b
    Fxf = max(fx * car.b / l, fx)
    Fxr = min(fx * car.a / l, 0.0)
    Fzf = car.mass * car.g * car.b / l
    Fzr = car.mass * car.g * car.a / l
    return Fxf, Fxr, Fzf, Fzr
end

function logit_lateral_force_model(car::BicycleCar, x)
    fx = x[2]
    r  = x[3]
    Uy = x[4]
    Ux = x[5]
    Cα = 1.0

    α_f = atan(Uy + car.a * r, Ux)
    α_r = atan(Uy - car.b * r, Ux)

    Fxf, Fxr, Fzf, Fzr = FWD_force_model(car, fx)
    Fyf = lateral_tire_model(car.μ, Cα, α_f, Fxf, Fzf)
    Fyr = lateral_tire_model(car.μ, Cα, α_r, Fxr, Fzr)
    return Fxf, Fxr, Fyf, Fyr
end

function lateral_tire_model(μ, Cα, α, Fx, Fz)
    Fy_max = sqrt((μ*Fz)^2 - Fx^2)
    slope = 2*Cα / Fy_max
    Fy = Fy_max * (1-2*exp(slope*α)) / (1 + exp(slope*α))
    return Fy
end


struct ContingencyCar{T} <: AbstractModel
    cars::Vector{BicycleCar{T}}
    ix::Vector{SVector{8,Int}}
    iu::Vector{SVector{2,Int}}
end

function ContingencyCar(num_cars=2)
    cars = [BicycleCar() for i = 1:num_cars]
    n,m = 8,2
    ix = [SVector{n}((1:n) .+ (i-1)*n) for i = 1:num_cars]
    iu = [SVector{m}((1:m) .+ (i-1)*m) for i = 1:num_cars]
    ContingencyCar(cars, ix, iu)
end


Base.size(::ContingencyCar) = 16,4

function dynamics(car::ContingencyCar, x, u, s)
    ix, iu = car.ix, car.iu
    cars = car.cars
    x1,u1 = x[ix[1]], u[iu[1]]
    ẋ1 = dynamics(cars[1], x1, u1, s)
    x2,u2 = x[ix[2]], u[iu[2]]
    ẋ2 = dynamics(cars[2], x1, u1, s)
    return [ẋ1; ẋ2]
end
