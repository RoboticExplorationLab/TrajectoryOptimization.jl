include("path.jl")
include("model.jl")
using TrajectoryOptimization
using TrajOptPlots
import TrajectoryOptimization: set_state!
using Plots

function gen_car_prob(U0, path, N; s0=0.0, L=10.0, δ0=0.0, integration=RK3)

	# Discretization
	sf = L + s0
	ds = (sf - s0)/(N-1)

	s_traj = range(s0,sf,step=ds)
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
	Ux0 = 15.
	Ux_des = 15.
	x0 = @SVector [δ0,     # δ
	               0,      # fx
	               0,      # r
	               0,      # Uy
	               Ux0, # Ux
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

	Qd = @SVector [0.000,      # δ
	               0.000,      # fx
	               0.00,      # r
	               0.00,      # Uy
	               0.1,      # Ux
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

	Rd   = @SVector [0.01,  # δ_dot
	                 1.0]    # fx_dot
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
	Fx_max = car.μ*car.mass*car.g / 1000  # friction cone?
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
	prob = Problem(car, obj, xd, sf, constraints=conSet, x0=x0, t0=s0,
		integration=integration)
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
line = StraightPath(30., pi/2)
arc = ArcPath(line, 20, 3pi/4)
path = DubinsPath([line, arc])
# path = line
plot(path, aspect_ratio=:equal)


N = 101
U0 = [@SVector [0,0] for k = 1:N-1]
prob = gen_car_prob(U0, path, N, s0=0.0, δ0=deg2rad(0), L=30.0, integration=RK3)

solver = iLQRSolver(prob)
solver = AugmentedLagrangianSolver(prob)
initial_controls!(solver, U_sol)
solver.opts.verbose = true
solver.opts.opts_uncon.verbose = true
solve!(solver)
plot(solver)
states(solver)[end][5]

cost(solver)
iterations(solver)
plot(controls(solver))
plot(states(solver), 1:2)
evaluate!(prob.constraints, get_trajectory(solver))
b = [v[1] for v in prob.constraints[2].vals]
plot(b)

plot(controls(prob))
plot(states(prob), 1:2)
plot(states(prob), 4:5)
plot(states(prob), 6:6)
plot(states(prob), 7:7)
plot(states(prob), 8:8)
states(prob)[end][5]
cost(prob)
TO.stage_cost(prob.obj.cost[10], states(prob)[10], controls(prob)[10])

Fx = states(prob.Z, 2)
Ux = states(prob.Z, 5)
dpsi = states(prob.Z, 6)
Fxf = [FWD_force_model(prob.model, fx*1e3)[1] for fx in Fx]
Fxr = [FWD_force_model(prob.model, fx*1e3)[2] for fx in Fx]
plot(Ux)
plot(Fxf)
plot(Fxr)

Uxdot = [dynamics(prob.model, z)[5] for z in prob.Z]
plot(Uxdot)


# Solve the initial problem
function initial_solve()
	N = 51
	U0 = [@SVector zeros(2) for k = 1:N-1]
	prob = gen_car_prob(U0, path, N, s0=0.0, δ0=deg2rad(0), L=30.0, integration=RK2)
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
	solver.opts.penalty_initial = 10.0
	solver.opts.penalty_scaling = 10.0
	solver.opts.cost_tolerance_intermediate = 1e-3
	solver.opts.constraint_tolerance = 1e-3

	# Set the new initial condition
	set_initial_state!(solver, state(Z[2]))

	nothing
end
solver = initial_solve()
TO.get_initial_state(solver)
states(solver)[end][5]
plot(solver)
plot(controls(solver))
findmax_violation(solver)

Nruns = 150
times = zeros(Nruns)
solver = initial_solve()
solver.opts.verbose = false
for i = 1:Nruns
	t = @elapsed begin
		advance_problem!(solver)
		solve!(solver)
	end
	s = get_trajectory(solver)[1].t
	println("run $i: iters=$(iterations(solver)), s=$s")
	times[i] = t
end
times
@show median(times)*1000
@show maximum(times)*1000
plot(solver)
plot(states(solver),7:7)

# advance_problem!(solver)
# U_sol = deepcopy(controls(solver))
#
# solver.opts.verbose = true
# solver.opts.reset_duals = true
# solver.opts.reset_penalties
# initial_controls!(solver, U_sol)
# solve!(solver)
# max_violation(solver)
# findmax_violation(solver)
# plot(solver)
# get_times(solver)[1]
# x0_sol = copy(TO.get_initial_state(solver))
#
# prob = gen_car_prob(U_sol, path, N, s0=get_times(solver)[1], δ0=deg2rad(0), L=30.0, integration=RK3)
# solver0 = AugmentedLagrangianSolver(prob)
# set_initial_state!(solver0, x0_sol)
# initial_controls!(solver0, U_sol)
# solver0.opts.verbose = true
# solve!(solver0)
#
# controls(solver) ≈ controls(solver0)
# TO.get_initial_state(solver) ≈ TO.get_initial_state(solver0)

#
# plot(path, aspect_ratio=:equal)
# e = [z.z[7] for z in Z]
# s = [z.t for z in Z]
# x,y = localToGlobal(path, e, s)

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
conSet = get_constraints(solver)
b = [v[1] for v in conSet.constraints[2].vals]
plot(b)
# #
# # plot(path.X,path.Y, aspect_ratio=:equal)
# # plot!(x,y)
plot(solver)
