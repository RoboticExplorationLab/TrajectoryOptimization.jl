include("model.jl")
import TrajectoryOptimization: set_state!

# Scenario
s = range(0,30,length=101)
k = fill(0.05,101)
k_itp = CubicSplineInterpolation(s,k)

# Discretization
sf = 3.
ds = 0.1

s_traj = range(0,sf,step=ds)
N = length(s_traj)
k_traj = k_itp.(s_traj)

# Model
car = BicycleCar()
n,m = size(car)
p_c = 0.1 # probability of contingency plan

# Set scenario
car.k_s = Dict(s_traj .=> k_traj)

# Objective
Ux_des = 5.
x0 = @SVector [0,      # δ
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
               0.1,      # fx
               0.1,      # r
               0.1,      # Uy
               0.5,      # Ux
               0.1,      # dpsi
               10.0,      # e
               0.1]      # t


Rd   = @SVector [0.01,  # δ_dot
                 0.01]    # fx_dot
Rd_c = @SVector [0.3282,  # δ_dot
                 4e-8]    # fx_dot

# Q = Diagonal([(1-p_c)*Qd; p_c*Qd])
# R = Diagonal([Rd; Rd_c])

# Qd = @SVector fill(0.1,n)
# Rd = @SVector fill(0.01,m)
Q = Diagonal(Qd)
R = Diagonal(Rd)

obj = LQRObjective(Q,R,Q*N,xd,N)

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

bnd = VariableBoundConstraint(n,m,N-1, x_max=x_max, x_min=x_min, u_min=u_min)

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
solver.opts.verbose = false
solve!(solver)
#
# @btime begin
# 	initial_controls!($solver, $U0)
# 	solve!($solver)
# end

# solve!(solver)
include("vis.jl")
lines(states(solver))
# Plots.plot(states(solver),4:6,label=labels(car)[:,4:6])

Plots.plot(states(solver),label=labels(car))

Plots.plot(states(solver), 7:7)

# Plots.plot(controls(solver))
