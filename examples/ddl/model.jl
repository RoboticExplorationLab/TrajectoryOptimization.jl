using Parameters, StaticArrays, BenchmarkTools, LinearAlgebra, Interpolations
# using MAT
using TrajectoryOptimization
import TrajectoryOptimization: dynamics, AbstractConstraint, evaluate, state_dim
const TO = TrajectoryOptimization




@with_kw mutable struct BicycleCar{T,P} <: AbstractModel
    path::P = StraightPath(30.0, 0.)
    a::T = 1.2169  # dist to front axle (m)
    b::T = 1.4131  # dist to rear axle (m)
    h::T = 0.5       # height of center of gravity? (m)
    mass::T = 1744.  # mass (kg)
    Iz::T = 2.7635e3  # moment of inertia (kg⋅m²)
    Cαf::T = 66000.  # (N/rad)
    Cαr::T = 60000.  # (N/rad)
    μ::T = 0.9       # coefficient of friction
    Cd0::T = 218.06  # Cd0 (N)
    Cd1::T = 0.0 # (N/mps)
    g::T = 9.81  # gravity (m/s²)
end

labels(::BicycleCar) = ["delta" "fx" "r" "Uy" "Ux" "dpsi" "e" "t"]

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
    k = curvature(car.path, s)

    # # Drag Force
    Fx_drag = 0.0
    #
    # # Spatial derivative
    s_dot = (Ux * cos(Δψ) - Uy * sin(Δψ)) / (1 - k*e)
    #
    # # Get tire forces
    Fxf, Fxr, Fyf, Fyr = logit_lateral_force_model(car, x)
    #
    # # State Derivatives
    r_dot  = (car.a * (Fyf * cos(δ) + Fxf*sin(δ)) - car.b * Fyr) / car.Iz
    Ux_dot =  r * Uy + (Fxf * cos(δ) - Fyf * sin(δ) + Fxr - Fx_drag) / car.mass
    Uy_dot = -r * Ux + (Fyf * cos(δ) + Fxf * sin(δ) + Fyr) / car.mass
    Δψ_dot = r - k * s_dot
    e_dot = Ux * sin(Δψ) + Uy * cos(Δψ)
    t_dot = 1 / s_dot


    # Slip angles
    αf = atan(Uy + car.a*r, Ux) - δ
    αr = atan(Uy - car.b*r, Ux)

    # # Get longitudinal forces
    # Fxf, Fxr, Fzf, Fzr = FWD_force_model(car, fx)

    # # Get lateral forces
    # Fyf = logit_tire_model(αf, car.μ, car.Cαf, Fxf, Fzf)
    # Fyr = logit_tire_model(αr, car.μ, car.Cαr, Fxr, Fzr)
    #
    # # Dynamics from paper
    # s_dot = Ux - Uy*Δψ
    # e_dot = Uy + Ux*Δψ
    #
    # Δψ_dot = r - k*Ux
    # Ux_dot = (Fxf + Fxr)/car.mass + r*Uy
    # Uy_dot = (Fyf + Fyr)/car.mass - r*Ux
    # r_dot = (car.a*Fyf - car.b*Fyr)/car.Iz
    # t_dot = 1/s_dot

    return t_dot * @SVector [δ_dot, fx_dot, r_dot, Uy_dot, Ux_dot, Δψ_dot, e_dot, 1]
end

function FWD_force_model(car::BicycleCar, fx)
    l = car.a + car.b
    Fxf = max(fx * car.b / l, fx)  # lower bound
    Fxr = min(fx * car.a / l, 0.0)  # lower bound
    # Fxf = fx*car.b / l
    # Fxr = fx*car.a / l
    Fzf = car.mass * car.g * car.b / l
    Fzr = car.mass * car.g * car.a / l
    return Fxf, Fxr, Fzf, Fzr
end

function logit_lateral_force_model(car::BicycleCar, x)
    δ = x[1]
    fx = x[2]
    r  = x[3]
    Uy = x[4]
    Ux = x[5]

    α_f = atan(Uy + car.a * r, Ux) - δ
    α_r = atan(Uy - car.b * r, Ux)

    Fxf, Fxr, Fzf, Fzr = FWD_force_model(car, fx)
    # Fyf = logit_tire_model(car.μ, car.Cαf, α_f, Fxf, Fzf)
    # Fyr = logit_tire_model(car.μ, car.Cαr, α_r, Fxr, Fzr)
    Fyf = fiala_tire_model(car.μ, car.Cαf, α_f, Fxf, Fzf)
    Fyr = fiala_tire_model(car.μ, car.Cαr, α_r, Fxr, Fzr)
    return Fxf, Fxr, Fyf, Fyr
end


function logit_tire_model(α, μ, Cα, Fx, Fz)
    Fy_max = sqrt((μ*Fz)^2 - Fx^2)
    slope = 2*Cα / Fy_max
    Fy = Fy_max * (1-2*exp(slope*α)) / (1 + exp(slope*α))
    return Fy
end

function fiala_tire_model(μ, Cα, α, Fx, Fz)
    Fx = min( 0.99*μ*Fz, Fx)  # try removing this?
    Fx = max(-0.99*μ*Fz, Fx)
    Fy_max = sqrt((μ*Fz)^2 - Fx^2)
    a_slide = atan(3*Fy_max, Cα)
    tan_a = tan(α)
    Fy_unsat = ( -Cα * tan_a + Cα^2/(3*Fy_max)*tan_a*abs(tan_a) - Cα^3/(27*Fy_max^2) * tan_a^3 )
    Fy_sat = -Fy_max*sign(α)
    if abs(α) < a_slide
        return Fy_unsat
    else
        return Fy_sat
    end
end

function load_scenario!(car::BicycleCar, s, k, s_interp)
    itp = CubicSplineInterpolation(s,k)
    k_interp = itp.(s_interp)
    car.k_c = Dict(s_interp .=> k_interp)
    return nothing
end

struct BrakeForceConstraint{L} <: AbstractConstraint{Inequality,State,1}
    car::L
end

state_dim(con::BrakeForceConstraint) = size(con.car)[1]

function evaluate(con::BrakeForceConstraint, x::SVector)
    car = con.car
    l = car.a + car.b

    Fx = x[2]
    r  = x[3]
    Uy = x[4]
    Ux = x[5]

    # Down force on front
    αf = atan(Uy + car.a*r, Ux)
    Fzf = car.mass*car.g*car.b/l

    # Long. force on front
    Fxf = Fx*car.b/l
    SVector{1}(-cos(αf)*car.μ*Fzf - Fxf)
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

function localToGlobal(path::AbstractPath, Z::Traj)
    e = [z.z[7] for z in Z]
    s = [z.t for z in Z]
    localToGlobal(path, s, e)
end

function Plots.plot(solver::TrajectoryOptimization.AbstractSolver)
    plot(get_model(solver), get_trajectory(solver))
end

function Plots.plot(car::BicycleCar, Z::Traj)
    path = car.path
    plot(path, aspect_ratio=:equal)
    x,y = localToGlobal(path, Z)
    plot!(x,y, linewidth=2)
end
