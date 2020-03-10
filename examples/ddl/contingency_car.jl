
include("path.jl")
include("model.jl")
using TrajectoryOptimization
using TrajOptPlots
import TrajectoryOptimization: set_state!
using Plots

function gen_car_prob(U0, path, N; s0=0.0, L=10.0, δ0=0.0, integration=RK3)

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

	Qd = @SVector [0.001,      # δ
	               0.001,      # fx
	               0.00,      # r
	               0.00,      # Uy
	               1.0,      # Ux
	               100.0,      # dpsi
	               100.0,      # e
	               0.0]      # t


	Qf = @SVector [0.000,      # δ
	               0.000,      # fx
	               0.000,      # r
	               0.000,      # Uy
	               10.0,      # Ux
	               100.0,      # dpsi
	               100.0,      # e
	               0.00]      # t

	Rd   = @SVector [0.1,  # δ_dot
	                 0.1]    # fx_dot
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
