using Plots
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO
using MeshIO
using LinearAlgebra
import TrajectoryOptimization: AbstractSolver, solve_aula!
const TO = TrajectoryOptimization

include("visualization.jl")
include("methods.jl")
include("models.jl")

# n = doubleintegrator2D.n
# m = doubleintegrator2D.m

car_model = Dynamics.car
car_model.info[:radius] = 1.0
n = Dynamics.car.n
m = Dynamics.car.m


N = 21
dt = 0.1

na = 5

function gen_x0(na,rad,cur_angle=0)
    angle = (pi)/na
    x0 = []
    for i = 1:na
        x = rad*cos(cur_angle)
        y = rad*sin(cur_angle)

        _x0 = [x;y;0;0]
        push!(x0,_x0)
        cur_angle += angle
    end
    x0
end

function gen_x0_car(na,rad,cur_angle=0)
    angle = (pi)/na
    x0 = []
    for i = 1:na
        x = rad*cos(cur_angle)
        y = rad*sin(cur_angle)

        _x0 = [x;y;cur_angle+pi]
        push!(x0,_x0)
        cur_angle += angle
    end
    x0
end

x0 = gen_x0(na,10)
xf = gen_x0(na,10,pi)

x0 = gen_x0_car(na,10)
xf = gen_x0_car(na,10,pi)

Q = [1.0*Diagonal(I,n) for i = 1:na]
Qf = [10.0*Diagonal(I,n) for i = 1:na]
R = [1.0e-1*Diagonal(I,m) for i = 1:na]

obj = [LQRObjective(Q[i],R[i],Qf[i],xf[i],N) for i = 1:na]

# Constraints
constraints = Constraints[]
for i = 1:na
    con = Constraints(N)
    # for k = 1:N-1
    #     con[k] +=
    # end
    con[N] += goal_constraint(xf[i])
    push!(constraints,copy(con))
end

U0 = [0.1*rand(m) for k = 1:N-1]

# create problems
probs = [Problem(doubleintegrator2D,
                obj[i],
                U0,
                integration=:midpoint,
                constraints=constraints[i],
                x0=x0[i],
                xf=xf[i],
                N=N,
                dt=dt)
                for i = 1:na]

probs = [Problem(car_model,
                obj[i],
                U0,
                integration=:midpoint,
                constraints=constraints[i],
                x0=x0[i],
                xf=xf[i],
                N=N,
                dt=dt)
                for i = 1:na]

# solver options
verbose=false

opts_ilqr = iLQRSolverOptions(verbose=verbose,
    iterations=500)

opts_al = AugmentedLagrangianSolverOptions{Float64}(verbose=verbose,
    opts_uncon=opts_ilqr,
    cost_tolerance=1.0e-6,
    constraint_tolerance=1.0e-3,
    cost_tolerance_intermediate=1.0e-5,
    iterations=100,
    penalty_scaling=2.0,
    penalty_initial=10.)

# Solve
@time probs_sol, solver_sol = solve_admm_collision(probs, :adaptive, opts_al,true)

cc

function plot_2D_traj(prob,p=plot(),xlim=(-15,15),ylim=(-15,15))
    x = Array([prob.X[k][1] for k = 1:prob.N])
    y = Array([prob.X[k][2] for k = 1:prob.N])
    p = plot!(x,y,xlim=xlim,ylim=ylim,marker=:circle,aspect_ratio=:equal,label="")
    p
end

# xy path
let
    p = plot()
    for i = 1:na
        p = plot_2D_traj(probs_sol[i],p)
    end
    display(p)
end

# x trajectories
let
    p = plot()
    t = range(0,stop=probs_sol[1].tf,length=probs_sol[1].N)
    for i = 1:na
        x = Array([probs_sol[i].X[k][1] for k = 1:probs_sol[i].N])
        plot!(t,x,xlabel="time (s)",ylabel="x")
    end
    display(p)
end

# x trajectories
let
    p = plot()
    t = range(0,stop=probs_sol[1].tf,length=probs_sol[1].N)
    for i = 1:na
        y = Array([probs_sol[i].X[k][2] for k = 1:probs_sol[i].N])
        plot!(t,y,xlabel="time (s)",ylabel="y")
    end
    display(p)
end
