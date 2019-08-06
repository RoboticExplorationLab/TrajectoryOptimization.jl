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

n_lift = doubleintegrator3D_lift.n
m_lift = doubleintegrator3D_lift.m

n_load = doubleintegrator3D_load.n
m_load = doubleintegrator3D_load.m

# Robot sizes
r_lift = 0.275
r_load = 0.2

# Control limits for lift robots
u_lim_u = Inf*ones(m_lift)
u_lim_u[1:3] .= 12/.850
u_lim_l = -Inf*ones(m_lift)
u_lim_l[3] = 0.

bnd = BoundConstraint(n_lift,m_lift,u_min=u_lim_l,u_max=u_lim_u)#,x_min=x_lim_lift_l)

# Obstacle constraints
r_cylinder = 0.5

_cyl = []
push!(_cyl,(5.,1.,r_cylinder))
push!(_cyl,(5.,-1.,r_cylinder))

function cI_cylinder_lift(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_lift)
    end
end
obs_lift = Constraint{Inequality}(cI_cylinder_lift,n_lift,m_lift,length(_cyl),:obs_lift)

function cI_cylinder_load(c,x,u)
    for i = 1:length(_cyl)
        c[i] = circle_constraint(x[1:3],_cyl[i][1],_cyl[i][2],_cyl[i][3] + 1.25*r_load)
    end
end
obs_load = Constraint{Inequality}(cI_cylinder_load,n_load,m_load,length(_cyl),:obs_load)

# initial state
scaling = 1.

shift_ = zeros(n_lift)
shift_[1:3] = [0.0;0.0;0.5]
x10 = zeros(n_lift)
x10[1:3] = scaling*[sqrt(8/9);0.;4/3]
x10 += shift_
x20 = zeros(n_lift)
x20[1:3] = scaling*[-sqrt(2/9);sqrt(2/3);4/3]
x20 += shift_
x30 = zeros(n_lift)
x30[1:3] = scaling*[-sqrt(2/9);-sqrt(2/3);4/3]
x30 += shift_
xload0 = zeros(n_load)
xload0[3] = 4/6
xload0 += shift_

xlift0 = [x10, x20, x30]

norm(xload0[1:3]-x10[1:3])
norm(xload0[1:3]-x20[1:3])
norm(xload0[1:3]-x30[1:3])
norm(x10[1:3]-x20[1:3])
norm(x20[1:3]-x30[1:3])
norm(x30[1:3]-x10[1:3])

# goal state
_shift = zeros(n_lift)
_shift[1:3] = [10.;0.0;0.0]

x1f = x10 + _shift
x2f = x20 + _shift
x3f = x30 + _shift
xloadf = xload0 + _shift

xliftf = [x1f, x2f, x3f]

norm(xloadf[1:3]-x1f[1:3])
norm(xloadf[1:3]-x2f[1:3])
norm(xloadf[1:3]-x3f[1:3])
norm(x1f[1:3]-x2f[1:3])
norm(x2f[1:3]-x3f[1:3])
norm(x3f[1:3]-x1f[1:3])

# Discretization
# N = 21
# dt = 0.1
#
# # Objective
# Q_lift = [1.0e-2*Diagonal(I,n_lift), 10.0e-2*Diagonal(I,n_lift), 0.1e-2*Diagonal(I,n_lift)]
# Qf_lift = [1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift)]
# R_lift = 1.0e-4*Diagonal(I,m_lift)
#
# Q_load = 0.0*Diagonal(I,n_load)
# Qf_load = 0.0*Diagonal(I,n_load)
# R_load = 1.0e-4*Diagonal(I,m_load)

N = 41
dt = 0.25

# Objective
Q_lift = [1.0e-2*Diagonal(I,n_lift), 1.0e-4*Diagonal(I,n_lift), 1.0e-2*Diagonal(I,n_lift)]
Qf_lift = [1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift),1.0*Diagonal(I,n_lift)]
R_lift = 1.0*Diagonal(I,m_lift)

Q_load = 0.0*Diagonal(I,n_load)
Qf_load = 0.0*Diagonal(I,n_load)
R_load = 1.0e-4*Diagonal(I,m_load)

obj_lift = [LQRObjective(Q_lift[i],R_lift,Qf_lift[i],xliftf[i],N) for i = 1:num_lift]
obj_load = LQRObjective(Q_load,R_load,Qf_load,xloadf,N)

# Constraints
constraints_lift = Constraints[]
for i = 1:num_lift
    con = Constraints(N)
    for k = 1:N-1
        con[k] += obs_lift + bnd
    end
    con[N] += goal_constraint(xliftf[i])
    push!(constraints_lift,copy(con))
end

constraints_load = Constraints(N)
for k = 1:N-1
    constraints_load[k] += obs_load
end
constraints_load[N] += goal_constraint(xloadf)


# initial control
u_ = [0.;0.;9.81 + 9.81/num_lift;0.;0.;-9.81/num_lift]
u_load = [0.;0.;9.81/num_lift;0.;0.;9.81/num_lift;0.;0.;9.81/num_lift]

U0_lift = [u_ for k = 1:N-1]
U0_load = [u_load for k = 1:N-1]

# create problems
prob_lift = [Problem(doubleintegrator3D_lift,
                obj_lift[i],
                U0_lift,
                integration=:midpoint,
                constraints=constraints_lift[i],
                x0=xlift0[i],
                xf=xliftf[i],
                N=N,
                dt=dt)
                for i = 1:num_lift]

prob_load = Problem(doubleintegrator3D_load,
                obj_load,
                U0_load,
                integration=:midpoint,
                constraints=constraints_load,
                x0=xload0,
                xf=xloadf,
                N=N,
                dt=dt)

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
@time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift,prob_load,n_slack,:parallel,opts_al)
# @time plift_al, pload_al, slift_al, sload_al = solve_admm(prob_lift,prob_load,n_slack,:sequential,opts_al)

# Visualize
vis = Visualizer()
open(vis)
visualize_DI_lift_system(vis,plift_al,pload_al,r_lift,r_load)


# xx = copy(plift_al[1].X)
# output_traj(plift_al[1])
