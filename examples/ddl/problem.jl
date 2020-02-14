include("path.jl")
include("model.jl")
using TrajectoryOptimization
using TrajOptPlots
import TrajectoryOptimization: set_state!
using Plots

function gen_car_prob(U0, path, N; s0=0.0, L=10.0, δ0=0.0)

	# Discretization
	sf = L + s0
	ds = sf/(N-1)

	s_traj = range(0,sf,step=ds)
	@assert N == length(s_traj)
	# k_traj = k_itp.(s_traj)
	# d_path = Dict(s_traj.=>k_traj)

	# Model
	car = BicycleCar(path=path)
	n,m = size(car)
	p_c = 0.1 # probability of contingency plan

	x,u = zeros(car)
	car.μ = 0.9
	fiala(a) = fiala_tire_model(car.μ, car.Cαf, a, 0.0, 10000)
	plot(fiala.(range(-0.2,0.2, length=100)))

	# Objective
	Ux_des = 15.
	x0 = @SVector [δ0,     # δ
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

	Qd = @SVector [0.01,      # δ
	               0.01,      # fx
	               0.01,      # r
	               0.01,      # Uy
	               1.0,      # Ux
	               10.0,      # dpsi
	               10.0,      # e
	               0.0]      # t


	Qf = @SVector [0.000,      # δ
	               0.000,      # fx
	               0.000,      # r
	               0.000,      # Uy
	               100.0,      # Ux
	               100.0,      # dpsi
	               100.0,      # e
	               0.00]      # t

	Rd   = @SVector [0.001,  # δ_dot
	                 0.001]    # fx_dot
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
	δ_dot_bound = deg2rad(50)  # deg/s
	δ_bound = deg2rad(27)  # deg
	Fx_max = car.μ*car.mass*car.g
	Ux_min = 1.0  # m/s
	e_bnd = 1.0 # m

	x_min = fill(-Inf,n)
	x_min[1] = -δ_bound
	x_min[5] = Ux_min
	x_min[7] = -e_bnd

	x_max = fill(Inf,n)
	x_max[1] = δ_bound
	x_max[2] = Fx_max
	x_max[7] = e_bnd

	u_min = [-δ_dot_bound; -Inf]
	u_max = [ δ_dot_bound;  Inf]

	bnd = BoundConstraint(n,m, x_max=x_max, x_min=x_min, u_min=u_min, u_max=u_max)

	# Brake constraint
	brake = BrakeForceConstraint(car)

	# Constraint Set
	conSet = ConstraintSet(n,m,N)
	add_constraint!(conSet, bnd, 1:N-1)
	add_constraint!(conSet, brake, 1:N)

	# Problem
	prob = Problem(car, obj, xd, sf, constraints=conSet, x0=x0, t0=s0)
	initial_controls!(prob, U0)

	return prob
end

# # Straight with circle
# t = range(0,20,length=901)
# radius = 20.
# X_curve = cos.(pi .- pi*t/30)*radius
# Y_curve = sin.(pi*t/30)*radius
# X_straight = fill(-radius, 100)
# Y_straight = collect(range(-5,0,length=100))
# X = [X_straight; X_curve]
# Y = [Y_straight; Y_curve]
#
# # Spiral
# t = range(0,20,length=901)
# radius = range(100,10,length=901)
# X = cos.(pi*t / 10) .* radius
# Y = sin.(pi*t / 10) .* radius
#
# s,ϕ,k = pathToLocal(X,Y)
# path = CarPath(s,ϕ,k,X,Y)
# plot(path.X,path.Y, aspect_ratio=:equal)
# s[end]

# Circle Path
line = StraightPath(10., pi/2)
arc = ArcPath(line, 30, pi/2)
path = DubinsPath([line, arc])
# path = line
plot(path, aspect_ratio=:equal)

# Solve the initial problem
function initial_solve()
	N = 101
	U0 = [@SVector zeros(2) for k = 1:N-1]
	prob = gen_car_prob(U0, path, N, s0=8.0, δ0=deg2rad(0), L=10.0)
	solver = AugmentedLagrangianSolver(prob)
	solver.opts.penalty_initial = 0.1
	solver.opts.opts_uncon.verbose = false
	solver.opts.verbose = true
	initial_controls!(solver, U0)
	rollout!(solver)
	solve!(solver)
end

function advance_problem!(solver)
	Z = get_trajectory(solver)

	# Advance the time and controls by one time step
	shift_fill!(Z)

	# Advance the duals
	solver.opts.reset_duals = false
	shift_fill!(get_constraints(solver))
	solver.opts.penalty_initial = 1000.0
	solver.opts.penalty_scaling = 100.0
	solver.opts.cost_tolerance_intermediate = 1e-2

	# Set the new initial condition
	set_initial_state!(solver, state(Z[2]))

	nothing
end
solver = initial_solve()
plot(solver)
findmax_violation(solver)

times = zeros(100)
solver = initial_solve()
solver.opts.verbose = false
for i = 1:100
	t = @elapsed begin
		advance_problem!(solver)
		solve!(solver)
	end
	times[i] = t
end
times
median(times)*1000
plot(solver)

plot(path, aspect_ratio=:equal)
e = [z.z[7] for z in Z]
s = [z.t for z in Z]
x,y = localToGlobal(path, e, s)

# # Simulate the next time step
# U_guess = U_sol[1:end]
# initial_controls!(prob, U_sol)
# solver2 = AugmentedLagrangianSolver(prob)
# rollout!(solver2)
# plot(solver2)
# solve!(solver)
# iterations(solver)
# plot(solver)
#
Z = get_trajectory(solver)
plot(states(Z),1:2)  # steering, accel
plot(states(Z),3:3)  # yaw rate
plot(states(Z),4:4)  # slip velocity
plot(states(Z),5:5)  # velocity
plot(states(Z),6:6)  # heading error
plot(states(Z),7:7)  # lateral error
plot(states(Z),8:8)  # time
plot(controls(Z))
# #
# # plot(path.X,path.Y, aspect_ratio=:equal)
# # plot!(x,y)
plot(solver)
