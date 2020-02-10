include("model.jl")
include("path.jl")
import TrajectoryOptimization: set_state!
using Plots

# Scenario
t = range(0,20,length=1001)
radius = 20
X = cos.(pi .- pi*t/30)*radius
Y = sin.(pi*t/30)*radius
s,ϕ,k = pathToLocal(X,Y)
path = CarPath(s,ϕ,k,X,Y)
plot(path.X,path.Y, aspect_ratio=:equal)
s[end]
nomX = interpolate((path.s,), path.X, Gridded(Linear()))
intp_heading = interpolate((path.s,), path.ϕ, Gridded(Linear()))


# Discretization
sf = 10.
ds = 0.05

s_traj = range(0,sf,step=ds)
N = length(s_traj)
# k_traj = k_itp.(s_traj)
# d_path = Dict(s_traj.=>k_traj)

# Model
car = BicycleCar(path=path)
n,m = size(car)
p_c = 0.1 # probability of contingency plan


# Objective
Ux_des = 5.
x0 = @SVector [deg2rad(0),      # δ
               0,      # fx
               0,      # r
               0,      # Uy
               Ux_des, # Ux
               0,      # dpsi
               0,      # e
               0]      # t

xd = @SVector [0,      # δ
               0,      # fx
               0,      # r
               0,      # Uy
               Ux_des, # Ux
               0,      # dpsi
               0,      # e
               0]      # t

Qd = @SVector [0.1,      # δ
               0.01,      # fx
               0.01,      # r
               0.01,      # Uy
               10.5,      # Ux
               0.1,      # dpsi
               10.0,      # e
               0.1]      # t


Qf = @SVector [0.001,      # δ
               0.001,      # fx
               0.001,      # r
               0.001,      # Uy
               1.0,      # Ux
               0.05,      # dpsi
               10.0,      # e
               0.00]      # t

Rd   = @SVector [0.10,  # δ_dot
                 0.01]    # fx_dot
Rd_c = @SVector [0.3282,  # δ_dot
                 4e-8]    # fx_dot

# Q = Diagonal([(1-p_c)*Qd; p_c*Qd])
# R = Diagonal([Rd; Rd_c])

# Qd = @SVector fill(0.1,n)
# Rd = @SVector fill(0.01,m)
Q = Diagonal(Qd)
R = Diagonal(Rd)

obj = LQRObjective(Q,R,Q,xd,N)

# Bound Constraints
δ_dot_bound = deg2rad(90)  # deg/s
δ_bound = deg2rad(27)  # deg
Fx_max = car.μ*car.mass*car.g
Ux_min = 1  # m/s


e_min = fill(-1, N)
e_max = fill( 1, N)

x_min = fill(-Inf,n)
x_min = map(1:N-1) do k
    x_min[1] = -δ_bound
    x_min[5] = Ux_min
    x_min[7] = e_min[k]
	x_min
end

x_max = fill(Inf,n)
x_max = map(1:N-1) do k
    x_max[1] = δ_bound
    x_max[2] = Fx_max
    x_max[7] = e_max[k]
	x_max
end

u_min = [[-δ_dot_bound; -Inf] for k = 1:N-1]
u_max = [[δ_dot_bound; Inf] for k = 1:N-1]

bnd = VariableBoundConstraint(n,m,N-1, x_max=x_max, x_min=x_min, u_min=u_min)
bnd = BoundConstraint(n,m, x_max=x_max[1], x_min=x_min[1], u_min=u_min[1], u_max=u_max[1])

# Brake constraint
brake = BrakeForceConstraint(car)

# Constraint Set
conSet = ConstraintSet(n,m,N)
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, brake, 1:N)

# Problem
prob = Problem(car, obj, xd, sf, constraints=conSet, x0=x0)
TO.set_times!(prob.Z, s_traj)
U0 = [@SVector zeros(m) for k = 1:N-1]

solver = iLQRSolver(prob)
solver = AugmentedLagrangianSolver(prob)
initial_controls!(solver, U0)
rollout!(solver)

solver.opts.verbose = true
solver.opts.opts_uncon.verbose = true
solve!(solver)
Z = get_trajectory(solver)
x,y = localToGlobal(path, Z)

rad2deg(state(Z[end])[6])
state(Z[end])[6]
solver.solver_uncon.xf
plot(states(Z),1:2)
plot(states(Z),3:3)  # yaw rate
plot(states(Z),4:4)  # slip velocity
plot(states(Z),5:5)  # velocity
plot(states(Z),6:6)  # heading error
plot(states(Z),7:7)  # lateral error
plot(states(Z),8:8)  # time

plot(controls(Z))

plot(path.X,path.Y, aspect_ratio=:equal)
plot!(x,y)
